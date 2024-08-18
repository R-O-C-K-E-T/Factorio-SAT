{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs";
    systems.url = "github:nix-systems/default-linux";
  };

  outputs = { self, nixpkgs, systems }:
    let
      inherit (nixpkgs) lib;

      eachSystem = lib.genAttrs (import systems);
      pkgsFor = eachSystem (system:
        import nixpkgs {
          localSystem.system = system;
          overlays = [ self.overlays.default ];
        });
    in {
      devShells = lib.mapAttrs (system: pkgs: {
        default = pkgs.mkShell {
          packages = [ pkgs.factorio-sat pkgs.factorio-sat-cli ];
        };
      }) pkgsFor;

      overlays = { default = import ./nix/overlay.nix; };

      packages = lib.mapAttrs (system: pkgs: {
        inherit (pkgs) factorio-sat factorio-sat-cli;
        default = self.packages.${system}.factorio-sat;
      }) pkgsFor;

      formatter =
        eachSystem (system: nixpkgs.legacyPackages.${system}.nixfmt-classic);
    };
}
