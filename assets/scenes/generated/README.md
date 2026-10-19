# Generated scenes

Built by `scripts/scenegen.py` (see `docs/scene_gen_pipeline.md`). Do not edit by hand:
`_manifest.jsonl` (seed + predicates + facts + plan) is canonical; USDA and task files are
regenerable build products. Thumbnails (after settle `--screenshot`): `_images/<scene>.png`.

- live scenes: **20**; attempts: 21; pre-GPU rejections by stage: {'hosted_drop': 1}; settle rejections: 0
- goal families: {'spatial': 3, 'stack': 3, 'put_on': 3, 'take_out': 3, 'put_in': 4, 'count_k_of_n': 2, 'reorient': 2}
- difficulty: {'simple': 13, 'moderate': 6, 'complex': 1}
- templates: {'blocks_stacking': 4, 'breakfast_plate': 4, 'desk_clutter_bin': 3, 'fruit_bowl': 3, 'mugs_reorient': 3, 'pantry_bins': 3}

| scene | family | difficulty | settled | instruction (default) |
| :-- | :-- | :-- | :-- | :-- |
| gen_blocks_stacking_0000 | spatial | simple | yes | Put the yellow block behind the red block |
| gen_blocks_stacking_0008 | stack | moderate | yes | Stack the blocks from bottom to top: red block and blue block |
| gen_blocks_stacking_0014 | stack | moderate | yes | Stack the blocks from bottom to top: yellow block and red block |
| gen_blocks_stacking_0020 | stack | moderate | yes | Stack the blocks from bottom to top: yellow block and red block |
| gen_breakfast_plate_0005 | spatial | simple | yes | Put the yogurt cup behind the bagel |
| gen_breakfast_plate_0007 | put_on | simple | yes | Put the banana on the plate |
| gen_breakfast_plate_0013 | put_on | simple | yes | Put the yogurt cup on the plate |
| gen_breakfast_plate_0019 | spatial | simple | yes | Put the cream cheese behind the milk carton |
| gen_desk_clutter_bin_0003 | take_out | simple | yes | Take the marker out of the large metal bin |
| gen_desk_clutter_bin_0011 | put_in | simple | yes | Put the marker in the green pail |
| gen_desk_clutter_bin_0017 | take_out | simple | yes | Take the phone out of the metal bin |
| gen_fruit_bowl_0006 | count_k_of_n | moderate | yes | Put exactly one piece of fruit in the wooden bowl |
| gen_fruit_bowl_0012 | take_out | simple | yes | Take the avocado out of the bowl |
| gen_fruit_bowl_0018 | count_k_of_n | complex | yes | Put exactly three fruits in the large black bowl |
| gen_mugs_reorient_0001 | reorient | moderate | yes | Set the mug upright |
| gen_mugs_reorient_0009 | put_on | simple | yes | Put the blue mug on the plate |
| gen_mugs_reorient_0015 | reorient | moderate | yes | Set the mug upright |
| gen_pantry_bins_0002 | put_in | simple | yes | Put the cream cheese in the metal bin |
| gen_pantry_bins_0010 | put_in | simple | yes | Put the small pail in the metal bin |
| gen_pantry_bins_0016 | put_in | simple | yes | Put the mayonnaise bottle in the large metal bin |
