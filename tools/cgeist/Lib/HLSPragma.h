#ifndef HLSPragma_H
#define HLSPragma_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "clang/Basic/SourceLocation.h"
#include <variant>

struct RegionPragma {
  RegionPragma(clang::SourceLocation loc) : loc(loc) {}
  clang::SourceLocation loc;
};

struct VarPragma {
  VarPragma(llvm::StringRef name) : name(name) {}
  std::string name;
};

struct ArgPragma {
  ArgPragma(llvm::StringRef funcName, int argNum)
      : funcName(funcName), argNum(argNum) {}
  std::string funcName;
  int argNum;
};

struct ResultPragma {
  ResultPragma(llvm::StringRef funcName, int resultNum)
      : funcName(funcName), resultNum(resultNum) {}
  std::string funcName;
  int resultNum;
};

typedef std::variant<RegionPragma, VarPragma, ArgPragma, ResultPragma>
    PragmaKind;

typedef std::variant<std::string, int> AttrKind;

struct HLSPragmaInfo {
  HLSPragmaInfo(PragmaKind pragmaKind,
                llvm::SmallVector<std::pair<std::string, AttrKind>, 4> attrs)

      : pragmaKind(pragmaKind), attrs(attrs), visited(false) {}
  PragmaKind pragmaKind;
  llvm::SmallVector<std::pair<std::string, AttrKind>, 4> attrs;
  bool visited;
};

struct HLSInfoList {
public:
  void addPragmaInfo(HLSPragmaInfo info) {
    if (std::holds_alternative<RegionPragma>(info.pragmaKind))
      regionInfoList.push_back(info);
    else if (std::holds_alternative<VarPragma>(info.pragmaKind))
      varInfoList.push_back(info);
    else if (std::holds_alternative<ArgPragma>(info.pragmaKind))
      argInfoList.push_back(info);
    else if (std::holds_alternative<ResultPragma>(info.pragmaKind))
      resultInfoList.push_back(info);
    else
      assert(false && "unreachable.");
  }

  llvm::SmallVector<std::pair<std::string, AttrKind>, 4>
  extractRegionPragmas(clang::SourceLocation beginLoc,
                       clang::SourceLocation endLoc) {
    llvm::SmallVector<std::pair<std::string, AttrKind>, 4> attrs;
    for (auto &info : regionInfoList) {
      if (info.visited)
        continue;
      auto loc = std::get<RegionPragma>(info.pragmaKind).loc;
      if (beginLoc < loc && loc < endLoc) {
        attrs.append(info.attrs);
        info.visited = true;
      }
    }
    return attrs;
  }

  llvm::SmallVector<std::pair<std::string, AttrKind>, 4>
  getVarPragmas(llvm::StringRef name) {
    llvm::SmallVector<std::pair<std::string, AttrKind>, 4> out;
    for (auto &info : varInfoList) {
      if (std::get<VarPragma>(info.pragmaKind).name == name)
        out.append(info.attrs);
    }
    return out;
  }

  llvm::SmallVector<std::pair<std::string, AttrKind>, 4>
  getArgPragmas(llvm::StringRef funcName, int argNum) {
    llvm::SmallVector<std::pair<std::string, AttrKind>, 4> out;
    for (auto &info : varInfoList) {
      auto argInfo = std::get<ArgPragma>(info.pragmaKind);
      if (argInfo.funcName == funcName && argInfo.argNum == argNum)
        out.append(info.attrs);
    }
    return out;
  }

  llvm::SmallVector<std::pair<std::string, AttrKind>, 4>
  getResultPragmas(llvm::StringRef funcName, int resultNum) {
    llvm::SmallVector<std::pair<std::string, AttrKind>, 4> out;
    for (auto &info : varInfoList) {
      auto resultInfo = std::get<ResultPragma>(info.pragmaKind);
      if (resultInfo.funcName == funcName && resultInfo.resultNum == resultNum)
        out.append(info.attrs);
    }
    return out;
  }

private:
  llvm::SmallVector<HLSPragmaInfo, 4> regionInfoList;
  llvm::SmallVector<HLSPragmaInfo, 4> varInfoList;
  llvm::SmallVector<HLSPragmaInfo, 4> argInfoList;
  llvm::SmallVector<HLSPragmaInfo, 4> resultInfoList;
};

static llvm::SmallVector<mlir::NamedAttribute, 4> getHLSNamedAttrs(
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
#endif