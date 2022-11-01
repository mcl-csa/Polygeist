//===- pragmaHandler.ch - Pragmas used to emit MLIR---------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRCLANG_LIB_PRAGMAHANDLER_H
#define MLIR_TOOLS_MLIRCLANG_LIB_PRAGMAHANDLER_H

#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Sema.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <clang/AST/Stmt.h>
#include <list>
#include <llvm/ADT/SmallString.h>
#include <variant>

/// POD holds information processed from the lower_to pragma.
struct LowerToInfo {
  llvm::StringMap<std::string> SymbolTable;
  llvm::SmallVector<llvm::StringRef, 2> InputSymbol;
  llvm::SmallVector<llvm::StringRef, 2> OutputSymbol;
};

/// The location of the scop, as delimited by scop and endscop
/// pragmas by the user.
/// "scop" and "endscop" are the source locations of the scop and
/// endscop pragmas.
/// "start_line" is the line number of the start position.
struct ScopLoc {
  ScopLoc() : end(0) {}

  clang::SourceLocation scop;
  clang::SourceLocation endscop;
  unsigned startLine;
  unsigned start;
  unsigned end;
};

/// Taken from pet.cc
/// List of pairs of #pragma scop and #pragma endscop locations.
struct ScopLocList {
  std::vector<ScopLoc> list;

  // Add a new start (#pragma scop) location to the list.
  // If the last #pragma scop did not have a matching
  // #pragma endscop then overwrite it.
  // "start" points to the location of the scop pragma.

  void addStart(clang::SourceManager &SM, clang::SourceLocation start) {
    ScopLoc loc;

    loc.scop = start;
    int line = SM.getExpansionLineNumber(start);
    start = SM.translateLineCol(SM.getFileID(start), line, 1);
    loc.startLine = line;
    loc.start = SM.getFileOffset(start);
    if (list.size() == 0 || list[list.size() - 1].end != 0)
      list.push_back(loc);
    else
      list[list.size() - 1] = loc;
  }

  // Set the end location (#pragma endscop) of the last pair
  // in the list.
  // If there is no such pair of if the end of that pair
  // is already set, then ignore the spurious #pragma endscop.
  // "end" points to the location of the endscop pragma.

  void addEnd(clang::SourceManager &SM, clang::SourceLocation end) {
    if (list.size() == 0 || list[list.size() - 1].end != 0)
      return;
    list[list.size() - 1].endscop = end;
    int line = SM.getExpansionLineNumber(end);
    end = SM.translateLineCol(SM.getFileID(end), line + 1, 1);
    list[list.size() - 1].end = SM.getFileOffset(end);
  }

  // Check if the current location is in the scop.
  bool isInScop(clang::SourceLocation target) {
    if (!list.size())
      return false;
    for (auto &scopLoc : list)
      if ((target >= scopLoc.scop) && (target <= scopLoc.endscop))
        return true;
    return false;
  }
};

enum PragmaKind {
  region, // Applies to the parent op of the region in which pragma is defined.
  named   // Applies to a named entity such as variable name.
};

struct HLSPragmaInfo {

  HLSPragmaInfo(clang::SourceLocation loc, PragmaKind pragmaKind,
                llvm::StringRef pragmaAttrName)
      : loc(loc), pragmaKind(pragmaKind), pragmaAttrName(pragmaAttrName) {}
  clang::SourceLocation loc;
  PragmaKind pragmaKind;
  std::string pragmaAttrName;
};

struct HLSUnrollInfo : public HLSPragmaInfo {
  HLSUnrollInfo(clang::SourceLocation loc, unsigned int factor)
      : HLSPragmaInfo(loc, PragmaKind::region, "HLS_UNROLL"), factor(factor) {}
  unsigned int factor;
};

struct HLSPipelineInfo : public HLSPragmaInfo {
  HLSPipelineInfo(clang::SourceLocation loc, unsigned int initiationInterval)
      : HLSPragmaInfo(loc, PragmaKind::region, "HLS_PIPELINE"),
        initiationInterval(initiationInterval) {}
  unsigned int initiationInterval;
};

struct HLSStorageInfo : public HLSPragmaInfo {
  HLSStorageInfo(clang::SourceLocation loc, llvm::StringRef name,
                 llvm::StringRef type, llvm::StringRef impl,
                 unsigned int latency)
      : HLSPragmaInfo(loc, PragmaKind::named, "HLS_STORAGE"), name(name),
        type(type), impl(impl), latency(latency) {}
  llvm::SmallString<4> name;
  llvm::SmallString<8> type;
  llvm::SmallString<8> impl;
  unsigned int latency;
};

struct HLSArrayPartitionInfo : public HLSPragmaInfo {
  HLSArrayPartitionInfo(clang::SourceLocation loc, llvm::StringRef name,
                        unsigned int partitionDim)
      : HLSPragmaInfo(loc, PragmaKind::named, "HLS_ARRAY_PARTITION"),
        name(name), partitionDim(partitionDim) {}
  llvm::SmallString<4> name;
  unsigned int partitionDim;
};

struct HLSExternFuncInfo : public HLSPragmaInfo {
  HLSExternFuncInfo(clang::SourceLocation loc, llvm::StringRef name,
                    unsigned int latency)
      : HLSPragmaInfo(loc, PragmaKind::named, "HLS_EXTERN_FUNC"), name(name),
        latency(latency) {}
  llvm::SmallString<4> name;
  unsigned int latency;
};

typedef std::variant<HLSUnrollInfo, HLSPipelineInfo, HLSStorageInfo,
                     HLSArrayPartitionInfo, HLSExternFuncInfo>
    HLSPragmaVariant;

struct HLSInfo {
  HLSInfo(HLSPragmaVariant v) : v(v), visited(false) {}
  HLSPragmaVariant v;
  bool visited;
  clang::SourceLocation getSrcLoc() {
    if (std::holds_alternative<HLSUnrollInfo>(v))
      return std::get<HLSUnrollInfo>(v).loc;
    if (std::holds_alternative<HLSPipelineInfo>(v))
      return std::get<HLSPipelineInfo>(v).loc;
    if (std::holds_alternative<HLSStorageInfo>(v))
      return std::get<HLSStorageInfo>(v).loc;
    if (std::holds_alternative<HLSArrayPartitionInfo>(v))
      return std::get<HLSArrayPartitionInfo>(v).loc;
    if (std::holds_alternative<HLSExternFuncInfo>(v))
      return std::get<HLSExternFuncInfo>(v).loc;
    assert(false && "Unreachable");
  }

  PragmaKind getPragmaKind() {
    if (std::holds_alternative<HLSUnrollInfo>(v))
      return std::get<HLSUnrollInfo>(v).pragmaKind;
    if (std::holds_alternative<HLSPipelineInfo>(v))
      return std::get<HLSPipelineInfo>(v).pragmaKind;
    if (std::holds_alternative<HLSStorageInfo>(v))
      return std::get<HLSStorageInfo>(v).pragmaKind;
    if (std::holds_alternative<HLSArrayPartitionInfo>(v))
      return std::get<HLSArrayPartitionInfo>(v).pragmaKind;
    if (std::holds_alternative<HLSExternFuncInfo>(v))
      return std::get<HLSExternFuncInfo>(v).pragmaKind;
    assert(false && "Unreachable");
  }

  llvm::StringRef getPragmaAttrName() {
    if (std::holds_alternative<HLSUnrollInfo>(v))
      return std::get<HLSUnrollInfo>(v).pragmaAttrName;
    if (std::holds_alternative<HLSPipelineInfo>(v))
      return std::get<HLSPipelineInfo>(v).pragmaAttrName;
    if (std::holds_alternative<HLSStorageInfo>(v))
      return std::get<HLSStorageInfo>(v).pragmaAttrName;
    if (std::holds_alternative<HLSArrayPartitionInfo>(v))
      return std::get<HLSArrayPartitionInfo>(v).pragmaAttrName;
    if (std::holds_alternative<HLSExternFuncInfo>(v))
      return std::get<HLSExternFuncInfo>(v).pragmaAttrName;
    assert(false && "Unreachable");
  }

  llvm::StringRef getName() {
    if (std::holds_alternative<HLSStorageInfo>(v))
      return std::get<HLSStorageInfo>(v).name;
    if (std::holds_alternative<HLSArrayPartitionInfo>(v))
      return std::get<HLSArrayPartitionInfo>(v).name;
    if (std::holds_alternative<HLSExternFuncInfo>(v))
      return std::get<HLSExternFuncInfo>(v).name;
    assert(false && "Unreachable");
  }
};

struct HLSInfoList {
public:
  void addPragmaInfo(HLSInfo info) { infoList.push_back(info); }

  llvm::SmallVector<HLSInfo, 4>
  extractRegionPragmas(clang::SourceLocation beginLoc,
                       clang::SourceLocation endLoc) {
    llvm::SmallVector<HLSInfo, 4> out;
    for (auto &info : infoList) {
      if (info.getPragmaKind() == PragmaKind::region && !info.visited &&
          info.getSrcLoc() > beginLoc && info.getSrcLoc() < endLoc) {
        out.push_back(info);
        info.visited = true;
      }
    }
    return out;
  }

  llvm::SmallVector<HLSInfo, 4> getNamedPragmas(llvm::StringRef name) {
    llvm::SmallVector<HLSInfo, 4> out;
    for (auto &info : infoList) {
      if (info.getPragmaKind() == PragmaKind::named && info.getName() == name) {
        out.push_back(info);
        info.visited = true;
      }
    }
    return out;
  }

private:
  llvm::SmallVector<HLSInfo, 4> infoList;
};

void addPragmaLowerToHandlers(clang::Preprocessor &PP, LowerToInfo &LTInfo);
void addPragmaScopHandlers(clang::Preprocessor &PP, ScopLocList &scopLocList);
void addPragmaEndScopHandlers(clang::Preprocessor &PP,
                              ScopLocList &scopLocList);
void addPragmaHLSHandler(clang::Preprocessor &PP, HLSInfoList &infoList);

#endif
