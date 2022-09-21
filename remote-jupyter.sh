ssh -N -o ProxyCommand="ssh -q s2259310@mlp.ed.ac.uk nc landonia01 22" \
  -L 8888:localhost:8888 s2259310@landonia01