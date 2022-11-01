#include "mlir/IR/Operation.h"
#include "clang/Basic/SourceLocation.h"

struct RegionPragma {
  RegionPragma(clang::SourceLocation loc) : loc(loc) {}
  clang::SourceLocation loc;
};

struct VarPragma {
  VarPragma(llvm::StringRef name) : name(name) {}
  std::string name;
};

struct ArgPragma {
  ArgPragma(llvm::StringRef funcName, llvm::StringRef argName)
      : funcName(funcName), argName(argName) {}
  std::string funcName;
  std::string argName;
};

typedef std::variant<RegionPragma, VarPragma, ArgPragma> PragmaKind;

struct HLSPragmaInfo {
  HLSPragmaInfo(clang::SourceLocation loc, PragmaKind pragmaKind,
                mlir::DictionaryAttr attrs)
      : loc(loc), pragmaKind(pragmaKind), attrs(attrs) {}
  clang::SourceLocation loc;
  PragmaKind pragmaKind;
  mlir::DictionaryAttr attrs;
};

void insertPragmaAttrs(mlir::Operation *operation) {}