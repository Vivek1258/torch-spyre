/*
 * Copyright 2026 The Torch-Spyre Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "job_plan.h"

#include <iostream>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "spyre_allocator.h"
#include "spyre_stream.h"
#include "spyre_tensor_impl.h"
#include "util/processSpyreCodeArtifacts.h"

namespace spyre {

void JobPlanStepH2D::construct(LaunchContext&,
                               const SpyreStream& stream) const {
  auto* params =
      flex::createDmaParams(host_address_, device_address_.total_size(),
                            /*to_device=*/true, &device_address_);
  params->pipeline_barrier = pipeline_barrier_;
  stream.launchH2D(params);
  flex::destroyDmaParams(params);
}

void JobPlanStepH2D::write(std::ostream& os) const {
  os << "  H2D (Host-to-Device)\n";
  os << "    Host address: " << host_address_ << "\n";
  os << "    Device CompositeAddress: " << device_address_ << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

void JobPlanStepD2H::construct(LaunchContext& ctx,
                               const SpyreStream& stream) const {
  if (std::holds_alternative<flex::CompositeAddress>(device_address_)) {
    const auto& device_address =
        std::get<flex::CompositeAddress>(device_address_);
    auto* params =
        flex::createDmaParams(host_address_, device_address.total_size(),
                              /*to_device=*/false, &device_address);
    params->pipeline_barrier = pipeline_barrier_;
    stream.launchD2H(params);
    flex::destroyDmaParams(params);
  } else {
    const uint64_t dmva = std::get<Dmva>(device_address_).value;
    auto segment_id = flex::dmvaToSegmentId(dmva);
    TORCH_CHECK(segment_id < ctx.inputs_outputs.size(),
                "D2H tensor-segment lookup out of range: segment ", segment_id,
                " but only ", ctx.inputs_outputs.size(),
                " launch args were provided");
    const auto& tensor = ctx.inputs_outputs.at(segment_id);
    const auto& tensor_address =
        static_cast<SharedOwnerCtx*>(tensor.storage().data_ptr().get_context())
            ->composite_addr;
    TORCH_CHECK(tensor_address.chunks().size() == 1,
                "Tensor address must have 1 chunk");
    const auto& base_chunk = tensor_address.chunks()[0];
    uint64_t segment_offset = dmva - (segment_id << flex::SEGMENT_SIZE_BITS);
    TORCH_CHECK(segment_offset + size_ <= tensor_address.total_size(),
                "D2H transfer out of bounds: offset ", segment_offset,
                " + size ", size_, " exceeds tensor allocation size ",
                tensor_address.total_size());
    flex::LogicalAddress offset_addr(base_chunk.addr.region_id,
                                     base_chunk.addr.offset + segment_offset);
    flex::Chunk offset_chunk(offset_addr, size_, base_chunk.domain_id);

    // Create shared_ptr to manage lifetime - will be kept alive by callback
    auto device_address =
        std::make_shared<flex::CompositeAddress>(offset_chunk);

    auto* params =
        flex::createDmaParams(host_address_, device_address->total_size(),
                              /*to_device=*/false, device_address.get());
    params->pipeline_barrier = pipeline_barrier_;
    params->callback = [device_address](void*) {};
    stream.launchD2H(params);
    flex::destroyDmaParams(params);
  }
}

void JobPlanStepD2H::write(std::ostream& os) const {
  os << "  D2H (Device-to-Host)\n";
  if (std::holds_alternative<flex::CompositeAddress>(device_address_)) {
    os << "    Device CompositeAddress: "
       << std::get<flex::CompositeAddress>(device_address_) << "\n";
  } else {
    os << "    Device dmva: " << std::get<Dmva>(device_address_).value << "\n";
  }
  os << "    Host address: " << host_address_ << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

void JobPlanStepCompute::construct(LaunchContext& ctx,
                                   const SpyreStream& stream) const {
  std::vector<const flex::CompositeAddress*> tensor_allocs;
  if (bind_io_addresses_) {
    for (auto& tensor : ctx.inputs_outputs) {
      flex::CompositeAddress* address =
          &(static_cast<SharedOwnerCtx*>(
                tensor.storage().data_ptr().get_context())
                ->composite_addr);
      tensor_allocs.push_back(address);
    }
  }
  auto* params = flex::createComputeParams(
      &program_address_, std::move(tensor_allocs), name_, bootstrap_offset_);
  params->pipeline_barrier = pipeline_barrier_;
  stream.launchCompute(params);
  flex::destroyComputeParams(params);
}

void JobPlanStepCompute::write(std::ostream& os) const {
  os << "  Device Compute\n";
  os << "    Name: " << (name_.empty() ? "(unnamed)" : name_) << "\n";
  os << "    Program CompositeAddress: " << program_address_ << "\n";
  os << "    Bind I/O addresses: " << (bind_io_addresses_ ? "yes" : "no")
     << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

void JobPlanStepHostCompute::construct(LaunchContext& ctx,
                                       const SpyreStream& stream) const {
  // Helper lambda to build HostCallbackParams and launch on the stream.
  // flex::RuntimeStream::launchOperationHostCallback() invokes the callback
  // synchronously in the calling thread, so exceptions propagate directly
  // through launchHostCallback() to the caller
  auto launch_host_callback = [this, &stream](auto&& callback) {
    auto* params = flex::createHostCallbackParams(
        std::forward<decltype(callback)>(callback), nullptr, pipeline_barrier_);
    // Use a scope-exit guard so params is freed even if launchHostCallback
    // throws (which it does when the synchronous host callback raises).
    struct Guard {
      flex::HostCallbackParams* p;
      ~Guard() {
        flex::destroyHostCallbackParams(p);
      }
    } guard{params};
    stream.launchHostCallback(params);
  };

  // Symbolic-shape dispatch (POC): the bundle declares extra input_arg slots
  // after the tensor base addresses -- the runtime dim value S and the per-core
  // count P = ceil(S / split). Build [base DMVAs...] + [S / P ...] in the order
  // codegen recorded (dims first, then per-core) and hand it to program
  // correction. This forces the address-forwarding path regardless of which
  // host-compute case would otherwise apply (the fake-symbol {0} path and the
  // extract-addresses path), and clears the DataConvertInfo input-count check
  // that expects one value per bundle func param. Guarded on a null
  // input_buffer_ so a real Case-1 pinned-buffer correction is never overridden
  // (the symbolic per-core correction always has input_buffer_ == nullptr).
  if (!symbolic_inputs_.empty() && input_buffer_ == nullptr) {
    std::vector<int64_t> addresses;
    addresses.reserve(ctx.inputs_outputs.size() + symbolic_inputs_.size());
    auto& allocator = SpyreAllocator::instance();
    for (auto& tensor : ctx.inputs_outputs) {
      addresses.push_back(allocator.compositeAddressToDmva(
          (static_cast<SharedOwnerCtx*>(tensor.storage().data_ptr().get_context())
               ->composite_addr)));
    }
    // Runtime symbolic size S from the max-reserved tensor: tensor.to("spyre",
    // max=N) records the symbolic dim index in reserved_dim, and resize_ keeps
    // the logical size at the concrete runtime value. POC assumes a single
    // symbolic dim, so one reserved tensor supplies S for every symbolic slot.
    int64_t s_val = -1;
    for (auto& tensor : ctx.inputs_outputs) {
      auto* impl = static_cast<SpyreTensorImpl*>(tensor.unsafeGetTensorImpl());
      if (impl->reserved_dim.has_value()) {
        s_val = tensor.size(*impl->reserved_dim);
        break;
      }
    }
    TORCH_CHECK(s_val >= 0,
                "symbolic dispatch: no max-reserved tensor found; the symbolic "
                "input must be created via tensor.to(\"spyre\", max=N)");
    for (const auto& si : symbolic_inputs_) {
      if (si.kind == SymbolicInput::Kind::Percore) {
        TORCH_CHECK(si.split > 0, "symbolic dispatch: invalid split ", si.split);
        addresses.push_back((s_val + si.split - 1) / si.split);  // ceil(S/split)
      } else {
        addresses.push_back(s_val);  // the symbolic dim value S
      }
    }
    launch_host_callback([this, addresses](void*) {
      deeptools::processComputeOnHostCommand(*hcm_, output_buffer_, &addresses);
    });
    return;
  }

  // Case 1: input_buffer_ is provided
  if (input_buffer_ != nullptr) {
    launch_host_callback([this](void*) {
      deeptools::processComputeOnHostCommand(*hcm_, output_buffer_,
                                             input_buffer_);
    });
    return;
  }

  // Case 2: fake symbols (ishape_ is {0})
  // Further discussion is required on "ishape". For now, it's vector<int64_t>,
  // and it's {0}, it's for fake symbols
  if (ishape_.size() == 1 && ishape_[0] == 0) {
    launch_host_callback([this](void*) {
      deeptools::processComputeOnHostCommand(*hcm_, output_buffer_, nullptr);
    });
    return;
  }

  // Case 3: extract addresses from context tensors
  std::vector<int64_t> addresses(ctx.inputs_outputs.size());
  int addr_idx = 0;
  auto& allocator = SpyreAllocator::instance();
  for (auto& tensor : ctx.inputs_outputs) {
    int64_t addr = allocator.compositeAddressToDmva(
        (static_cast<SharedOwnerCtx*>(tensor.storage().data_ptr().get_context())
             ->composite_addr));
    addresses[addr_idx++] = addr;
  }

  launch_host_callback([this, addresses](void*) {
    deeptools::processComputeOnHostCommand(*hcm_, output_buffer_, &addresses);
  });
}

void JobPlanStepHostCompute::write(std::ostream& os) const {
  os << "  Host Compute\n";
  os << "    Output buffer: " << output_buffer_ << "\n";
  os << "    HCM metadata: " << (hcm_ ? "present" : "null") << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

std::ostream& operator<<(std::ostream& os, const JobPlan& plan) {
  os << "============ JobPlan =============\n";
  os << "Total steps: " << plan.steps.size() << "\n";

  // Job allocation
  size_t addr_idx = 0;
  for (const auto& addr : plan.job_allocation) {
    if (addr_idx == 0) {
      os << "Job allocation: " << addr << "\n";
    } else {
      os << "Program " << addr_idx - 1 << ": " << addr << "\n";
    }
    ++addr_idx;
  }

  // Expected input shapes
  if (!plan.expected_input_shapes.empty()) {
    os << "Expected input shapes (" << plan.expected_input_shapes.size()
       << " tensors):\n";
    for (size_t i = 0; i < plan.expected_input_shapes.size(); ++i) {
      os << "  Input " << i << ": [";
      for (size_t j = 0; j < plan.expected_input_shapes[i].size(); ++j) {
        if (j > 0) os << ", ";
        os << plan.expected_input_shapes[i][j];
      }
      os << "]\n";
    }
  }

  // Pinned buffers
  os << "Pinned buffers: " << plan.pinned_buffers.size() << "\n";
  for (size_t i = 0; i < plan.pinned_buffers.size(); ++i) {
    const auto& buf = plan.pinned_buffers[i];
    os << "  Buffer " << i << ": ptr=" << buf.data() << ", size=" << buf.size()
       << " bytes\n";
  }

  // Detailed step information
  os << "\nDetailed Steps:\n";
  for (size_t i = 0; i < plan.steps.size(); ++i) {
    os << "Step " << i << ": ";
    os << *plan.steps[i];
  }

  os << "==================================\n";
  return os;
}

}  // namespace spyre
