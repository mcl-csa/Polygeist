#include "HLSPragma.h"
llvm::SmallVector<mlir::NamedAttribute, 4> getHLSNamedAttrs(
    mlir::Builder &builder,
    llvm::SmallVectorImpl<std::pair<std::string, AttrKind>> &attrs) {
  llvm::SmallVector<mlir::NamedAttribute, 4> attrList;
  for (auto kv : attrs) {
    mlir::Attribute attr;
    if (std::holds_alternative<std::string>(kv.second))
      attr = builder.getStringAttr(std::get<std::string>(kv.second));
    else if (std::holds_alternative<int>(kv.second))
      attr = builder.getI64IntegerAttr(std::get<int>(kv.second));
    else
      assert(false && "unreachable.");
    attrList.push_back(builder.getNamedAttr(kv.first, attr));
  }
  return attrList;
}