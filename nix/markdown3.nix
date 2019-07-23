{ markdown, fetchPypi }:

markdown.overridePythonAttrs rec {
  pname = "Markdown";
  version = "3.1.1";
  src = fetchPypi {
    inherit pname version;
    sha256 = "0yhylk4ffqqs7x086fav4pnfsl1021v7lghznzkififprmmqfl1f";
  };
}
