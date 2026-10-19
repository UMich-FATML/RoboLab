# Scene templates

Hand-authored, committed scene templates for the generator (`scripts/scenegen.py`,
see `docs/scene_gen_pipeline.md`). A template names object *classes* and
*relations* with count ranges; the seeded sampler instantiates it, and mesh
geometry decides what physically fits where. Future LLM-authored templates use
this same schema.

```yaml
name: fruit_bowl                 # unique; scene files are gen_<name>_<seed>.usda
theme: one-line description
slots:
  - slot: bowl                   # slot id, referenced by in:/on: placements
    select: {category: bowl}     # category (str|list) / names (list) / color
    require: {container: true}   # geometry roles: container / support / graspable
    count: [1, 1]
    placement: table             # table | in:<slot> | on:<slot>
    region: center               # table only: center|left|right|front|back|any
    pose: upright                # upright | lying | any
goals: [put_in, take_out]        # goal families quest.py may sample
```
