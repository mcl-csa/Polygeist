//===- pragmaHandler.cc - Pragmas used to emit MLIR--------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pragmaHandler.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Attr.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Lex/LiteralSupport.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/ParseDiagnostic.h"
#include "clang/Sema/Sema.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <clang/Basic/DiagnosticLex.h>
#include <clang/Basic/DiagnosticParse.h>
#include <variant>

using namespace clang;
using namespace llvm;

namespace {

/// Handles the #pragma lower_to(<identifier>, "<mlir function target>")
/// directive.
class PragmaLowerToHandler : public PragmaHandler {
  LowerToInfo &Info;

public:
  PragmaLowerToHandler(LowerToInfo &Info)
      : PragmaHandler("lower_to"), Info(Info) {}

  /// Handle (x,y,z) segment.
  bool ReadInputOrOutput(Preprocessor &PP, Token &CurrentTok,
                         SmallVectorImpl<StringRef> &Ids) {
    PP.Lex(CurrentTok);
    if (CurrentTok.isNot(tok::l_paren)) {
      PP.Diag(CurrentTok.getLocation(), diag::warn_pragma_expected_lparen)
          << "lower_to";
      return false;
    }
    PP.Lex(CurrentTok);
    while (CurrentTok.isNot(tok::r_paren) && CurrentTok.isNot(tok::eod)) {
      if (CurrentTok.isNot(tok::identifier)) {
        PP.Diag(CurrentTok.getLocation(), diag::warn_pragma_expected_identifier)
            << "lower_to";
        return false;
      } else {
        StringRef Id = CurrentTok.getIdentifierInfo()->getName();
        Ids.push_back(Id);
      }
      PP.Lex(CurrentTok);
      if (CurrentTok.is(tok::r_paren))
        break;
      if (CurrentTok.isNot(tok::comma)) {
        PP.Diag(CurrentTok.getLocation(), diag::warn_pragma_expected_comma)
            << "lower_to";
        return false;
      }
      PP.Lex(CurrentTok);
    }
    // move to the next Token after l_paren.
    PP.Lex(CurrentTok);
    return true;
  }

  /// Handle input(a,b,c), output(x, y, z) optional segment.
  bool HandleOptionalInputAndOutput(Preprocessor &PP, Token &PragmaTok,
                                    SmallVectorImpl<StringRef> &Inputs,
                                    SmallVectorImpl<StringRef> &Outputs) {
    Token CurrentTok;
    PP.Lex(CurrentTok);

    // early exit.
    if (CurrentTok.is(tok::eod))
      return true;

    if (CurrentTok.isNot(tok::string_literal) ||
        (StringRef(CurrentTok.getLiteralData(), CurrentTok.getLength())
             .compare("input") == 0)) {
      PP.Diag(CurrentTok.getLocation(),
              diag::warn_pragma_expected_section_label_or_name)
          << "";
      return false;
    } else {
      if (!ReadInputOrOutput(PP, CurrentTok, Inputs))
        return false;
    }

    // comma.
    if (CurrentTok.isNot(tok::comma)) {
      PP.Diag(CurrentTok.getLocation(), diag::warn_pragma_expected_comma)
          << "lower_to";
      return false;
    }
    // move to the next Token after comma.
    PP.Lex(CurrentTok);

    if (CurrentTok.isNot(tok::string_literal) ||
        (StringRef(CurrentTok.getLiteralData(), CurrentTok.getLength())
             .compare("output") == 0)) {
      PP.Diag(CurrentTok.getLocation(),
              diag::warn_pragma_expected_section_label_or_name)
          << "expect 'output' for lower to";
      return false;
    } else {
      if (!ReadInputOrOutput(PP, CurrentTok, Outputs))
        return false;
    }

    if (CurrentTok.isNot(tok::eod)) {
      PP.Diag(CurrentTok.getLocation(), diag::warn_pragma_extra_tokens_at_eol)
          << "lower_to";
      return false;
    }

    return true;
  }

  /// The pragma handler will extract the single argument to the lower_to(...)
  /// pragma definition, which is the target MLIR function symbol, and relate
  /// the function decl that lower_to is attached to with that MLIR function
  /// symbol in the class-referenced dictionary.
  ///
  // #pragma lower_to(copy_op, "memref.copy") input(a), output(b)
  void HandlePragma(Preprocessor &PP, PragmaIntroducer Introducer,
                    Token &PragmaTok) override {
    Token Tok;
    PP.Lex(Tok); // pragma lparen
    if (Tok.isNot(tok::l_paren)) {
      PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_lparen)
          << "lower_to";
      return;
    }

    Token PrevTok = Tok;
    llvm::StringRef FuncId = llvm::StringRef();
    llvm::StringRef SymbolName = llvm::StringRef();
    while (Tok.isNot(tok::eod)) {
      Token CurrentTok;
      PP.Lex(CurrentTok);

      // pragma rparen.
      if (PrevTok.is(tok::string_literal)) {
        if (CurrentTok.isNot(tok::r_paren)) {
          PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_rparen)
              << "lower_to";
          return;
        } else {
          if (!HandleOptionalInputAndOutput(PP, CurrentTok, Info.InputSymbol,
                                            Info.OutputSymbol))
            return;
          else
            break;
        }
      }

      // function identifier.
      if (PrevTok.is(tok::l_paren)) {
        if (CurrentTok.isNot(tok::identifier)) {
          PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_identifier)
              << "lower_to";
          return;
        } else {
          FuncId = CurrentTok.getIdentifierInfo()->getName();
        }
      }

      // comma.
      if (PrevTok.is(tok::identifier)) {
        if (CurrentTok.isNot(tok::comma)) {
          PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_comma)
              << "lower_to";
          return;
        }
      }

      // string literal, which is the MLIR function symbol.
      if (PrevTok.is(tok::comma)) {
        if (CurrentTok.isNot(tok::string_literal)) {
          PP.Diag(CurrentTok.getLocation(),
                  diag::warn_pragma_expected_section_name)
              << "lower to";
          return;
        } else {
          SmallVector<Token, 1> SymbolToks;
          SymbolToks.push_back(CurrentTok);
          SymbolName = StringLiteralParser(SymbolToks, PP).GetString();
        }
      }

      PrevTok = CurrentTok;
    }

    // Link SymbolName with the function.
    auto result = Info.SymbolTable.try_emplace(FuncId, SymbolName);
    assert(result.second &&
           "Shouldn't define lower_to over the same func id more than once.");
  }

private:
};

struct PragmaScopHandler : public PragmaHandler {
  ScopLocList &scops;

  PragmaScopHandler(ScopLocList &scops) : PragmaHandler("scop"), scops(scops) {}

  void HandlePragma(Preprocessor &PP, PragmaIntroducer Introducer,
                    Token &scopTok) override {
    auto &SM = PP.getSourceManager();
    auto loc = scopTok.getLocation();
    scops.addStart(SM, loc);
  }
};

struct PragmaEndScopHandler : public PragmaHandler {
  ScopLocList &scops;

  PragmaEndScopHandler(ScopLocList &scops)
      : PragmaHandler("endscop"), scops(scops) {}

  void HandlePragma(Preprocessor &PP, PragmaIntroducer introducer,
                    Token &endScopTok) override {
    auto &SM = PP.getSourceManager();
    auto loc = endScopTok.getLocation();
    scops.addEnd(SM, loc);
  }
};
SmallVector<std::pair<std::string, AttrKind>, 4>
parsePragmaAttrs(clang::Preprocessor &PP, clang::Token &currentTok,
                 std::string &prepend) {
  SmallVector<std::pair<std::string, AttrKind>, 4> out;
  while (true) {
    PP.Lex(currentTok); // key
    if (!currentTok.isAnyIdentifier())
      break;
    std::string key =
        prepend + currentTok.getIdentifierInfo()->getName().upper();
    PP.Lex(currentTok); //=
    assert(currentTok.is(tok::TokenKind::equal));
    PP.Lex(currentTok); // value
    AttrKind value;
    if (currentTok.isLiteral()) {
      auto s = std::string(currentTok.getLiteralData());
      value = std::stoi(s);
    } else if (currentTok.isAnyIdentifier()) {
      auto s = currentTok.getIdentifierInfo()->getName().str();
      value = s;
    } else {
      assert(false && "unreachable.");
    }
    out.push_back(std::make_pair(key, value));
  }
  return out;
}

struct PragmaHLSHandler : public PragmaHandler {
  HLSInfoList &infoList;
  PragmaHLSHandler(HLSInfoList &infoList)
      : PragmaHandler("HLS"), infoList(infoList) {}
  void HandlePragma(Preprocessor &PP, PragmaIntroducer introducer,
                    Token &hlsTok) override {

    auto loc = hlsTok.getLocation();

    PP.Lex(hlsTok);
    auto pragmaName = hlsTok.getIdentifierInfo()->getName().upper();
    std::string prepend = "hls." + pragmaName + "_";
    SmallVector<std::pair<std::string, AttrKind>, 4> attrs =
        parsePragmaAttrs(PP, hlsTok, prepend);
    if (pragmaName == "UNROLL" || pragmaName == "PIPELINE") {
      HLSPragmaInfo info(RegionPragma(loc), attrs);
      infoList.addPragmaInfo(info);
    } else if (pragmaName == "BIND_STORAGE" ||
               pragmaName == "ARRAY_PARTITION" || pragmaName == "EXTERN_FUNC") {
      Optional<std::string> name;
      for (auto kv : attrs) {
        if (kv.first == prepend + "VARIABLE" &&
            std::holds_alternative<std::string>(kv.second))
          name = std::get<std::string>(kv.second);
      }
      if (!name.has_value())
        PP.Diag(hlsTok, diag::err_expected) << " variable=<variable name>";

      HLSPragmaInfo info(VarPragma(*name), attrs);
      infoList.addPragmaInfo(info);
    } else if (pragmaName == "INTERFACE") {
      Optional<std::string> name;
      for (auto kv : attrs) {
        if (kv.first == prepend + "PORT" &&
            std::holds_alternative<std::string>(kv.second))
          name = std::get<std::string>(kv.second);
      }
      if (!name.has_value())
        PP.Diag(hlsTok, diag::err_expected) << " port=<port name>";

      HLSPragmaInfo info(VarPragma(*name), attrs);
      infoList.addPragmaInfo(info);
    } else {
      PP.Diag(hlsTok.getLocation(), diag::warn_pragma_expected_identifier)
          << "HLS UNROLL | PIPELINE | BIND_STORAGE | ARRAY_PARTITION | "
             "INTERFACE | EXTERN_FUNC.";
    }
  }
};
} // namespace

void addPragmaLowerToHandlers(Preprocessor &PP, LowerToInfo &LTInfo) {
  PP.AddPragmaHandler(new PragmaLowerToHandler(LTInfo));
}

void addPragmaScopHandlers(Preprocessor &PP, ScopLocList &scopLocList) {
  PP.AddPragmaHandler(new PragmaScopHandler(scopLocList));
}

void addPragmaEndScopHandlers(Preprocessor &PP, ScopLocList &scopLocList) {
  PP.AddPragmaHandler(new PragmaEndScopHandler(scopLocList));
}

void addPragmaHLSHandler(clang::Preprocessor &PP, HLSInfoList &infoList) {
  PP.AddPragmaHandler(new PragmaHLSHandler(infoList));
}