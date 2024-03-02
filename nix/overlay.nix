pkgs: pkgs0:
let
  inherit (pkgs) lib;
  antlr4_7_2 = (pkgs.callPackage ./antlr { })."4.7.2";
  python = pkgs.python3.override {
    self = python;
    packageOverrides = lib.composeManyExtensions [
      antlr4_7_2.pythonOverrides
      (pypkgs: _: {
        luaparser =
          pypkgs.toPythonModule (pypkgs.callPackage ./luaparser.nix { });
      })
    ];
  };
in {
  factorio-sat = python.pkgs.callPackage ./factorio-sat.nix { };
  factorio-sat-cli = pkgs.callPackage ./cli.nix { };
}
