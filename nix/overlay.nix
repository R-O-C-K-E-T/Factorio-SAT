{ self }:
final: _:
let pkgs = final;
in let
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

  mkDate = longDate:
    (lib.concatStringsSep "-" [
      (builtins.substring 0 4 longDate)
      (builtins.substring 4 2 longDate)
      (builtins.substring 6 2 longDate)
    ]);
  date = mkDate (self.lastModifiedDate or "19700101");
  rev = self.shortRev or "dirty";
in {
  factorio-sat = python.pkgs.callPackage ./factorio-sat.nix {
    versionSuffix = "+date=${date}_${rev}";
  };
  factorio-sat-cli = pkgs.callPackage ./cli.nix { };
}
