// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/pir/transforms/xpu/add_activation_fuse_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {
std::set<std::string> act_ops = {{paddle::dialect::ReluOp::name()},
                                 {paddle::dialect::GeluOp::name()}};

class AddActivationPattern : public paddle::drr::DrrPatternBase {
 private:
  std::string act_type_;
 public:
  AddActivationPattern(const std::string &act_type)
      : act_type_(act_type) {}
  std::string name() const override { return "AddActivationPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    
    const auto &add = pat.Op(paddle::dialect::AddOp::name());

    const auto &act = pat.Op(act_type_);
    add({&pat.Tensor("add_x"), &pat.Tensor("add_y")}, {&pat.Tensor("add_out")});
    pat.Tensor("act_out") = act(pat.Tensor("add_out"));

    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      if (pir::ValueIsPersistable(match_ctx.Tensor("add_x")) ||
          pir::ValueIsPersistable(match_ctx.Tensor("add_y"))) {
        return false;
      }
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &add_act_xpu =
        res.Op(paddle::dialect::AddActXpuOp::name());
    add_act_xpu({&res.Tensor("x"),
                 &res.Tensor("y")},
                {&res.Tensor("act_out"),
                 &res.Tensor("out_max")});
  }
};

class AddActivationXpuFusePass : public pir::PatternRewritePass {
 public:
  AddActivationXpuFusePass()
      : pir::PatternRewritePass("add_activation_xpu_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    for (auto act_op : act_ops) {
      ps.Add(paddle::drr::Create<AddActivationPattern>(
          context,
          act_op));
    }
    return ps;
  }
};

}  // namespace

namespace pir {
std::unique_ptr<Pass> CreateAddActivationXpuFusePass() {
  return std::make_unique<AddActivationXpuFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(add_activation_xpu_fuse_pass, AddActivationXpuFusePass);
