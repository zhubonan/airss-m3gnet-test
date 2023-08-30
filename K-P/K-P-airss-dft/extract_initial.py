from pathlib import Path

import tqdm
from matador.export import doc2cell
from matador.scrapers.castep_scrapers import castep2dict

if __name__ == "__main__":
    data_dir = Path("castep_files")
    castep_files = data_dir.glob("*.castep")
    for castep_file in tqdm.tqdm(castep_files):
        if Path(castep_file.stem + "-orig.cell").is_file():
            continue
        try:
            c, s = castep2dict(str(castep_file), db=True, intermediates=True)
            doc2cell(c["intermediates"][0], castep_file.stem + "-orig.cell")
        except:
            pass
