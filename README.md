<h1 align="center">COCO Mountains</h1>
<center>
Generate 3D landscapes from COCO functions.
</center>

## Installation

1. Create the environment
   ~~~sh
   micromamba create -y -f env.yaml
   ~~~
1. Run `coco-mountains.py` (takes about 3 minutes and generates about 400 MB of data)
  ~~~sh
  micromamba run -n coco-mountains python coco-mountains.py
  ~~~

You now have a bunch of STL files in the current directory. Have fun!
