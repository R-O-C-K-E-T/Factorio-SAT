{ lib, writeShellApplication, factorio-sat, }:
let
  exes = lib.pipe (builtins.readDir "${factorio-sat}/bin") [
    (lib.filterAttrs
      (name: kind: kind == "regular" && !(lib.hasPrefix "." name)))
    (lib.mapAttrsToList (name: _: name))
  ];
in writeShellApplication {
  name = "factorio-sat";
  text = ''
    show_cmds() {
      echo '${lib.concatStringsSep ", " exes}'
    }

    if [[ $# -eq 0 ]]; then
      echo 'Need subcommand.'
      echo
      show_cmds
    fi

    exe="$1"
    shift 1
    case "$exe" in
      ${
        lib.concatMapStrings (exe: ''
          ${exe})
            ${lib.getExe' factorio-sat exe} "$@"
            ;;
        '') exes
      }
      *)
        echo "No such subcommand: $exe"
        echo
        show_cmds
        exit 1
    esac
  '';
}
