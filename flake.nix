{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs";
    systems.url = "github:nix-systems/default-linux";
    nixfmt.url = "github:serokell/nixfmt";
  };

  outputs = { self, nixpkgs, systems, nixfmt }:
    let
      inherit (nixpkgs) lib;

      eachSystem = lib.genAttrs (import systems);
      pkgsFor =
        eachSystem (system: import nixpkgs { localSystem.system = system; });
    in {
      packages = eachSystem (system:
        let pkgs = pkgsFor.${system};
        in pkgs.callPackage ./nix/packages.nix { } // {
          default = self.packages.${system}.factorio-sat;
        });

      formatter = eachSystem (system: nixfmt.packages.${system}.default);
    };
}
