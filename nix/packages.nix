{ lib, callPackage, python3 }:
let
  antlr4_7_2 = (callPackage ./antlr { })."4.7.2";

  python = python3.override {
    self = python;
    packageOverrides = lib.composeManyExtensions [
      antlr4_7_2.pythonOverrides
      (pypkgs: _: {
        luaparser = pypkgs.toPythonModule (pypkgs.callPackage ./luaparser.nix { });
      })
    ];
  };

  factorio-sat = python.pkgs.callPackage ./factorio-sat.nix { };
in {
  inherit factorio-sat;
}
