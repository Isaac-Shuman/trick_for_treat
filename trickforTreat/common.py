"""
trickforTreat/common.py  (PSEUDOCODE, shared module)

Small helpers used by every script:
  load_config(path) -> attribute-style dict (yaml.safe_load + a tiny namespace wrapper);
                       apply CLI overrides like --limit; assert required keys exist.
  set_seed(seed)    -> random.seed, np.random.seed, torch.manual_seed, torch.cuda.manual_seed_all,
                       os.environ["PYTHONHASHSEED"]
  seed_worker(id)   -> DataLoader worker_init_fn (seeds numpy/random per worker from torch seed)
  pick_device(cfg)  -> "cuda" if available, else "mps", else "cpu" (unless config forces one)
  read_csv / write_csv  -> csv.DictReader/DictWriter with fixed column order, sorted by image_id
  safe_filename(s)  -> "lying down" -> "lying_down" for folder names
  setup_logging(cfg)
  save_contact_sheet(images, captions, path, cols=8)  -> debug grids used by 01, 04, 05
"""

# trickforTreat/transforms.py (planned) holds build_train_transform(cfg) (see 05 for the spec),
# kept separate from classifier.py because the live pipeline never needs training augmentations.
