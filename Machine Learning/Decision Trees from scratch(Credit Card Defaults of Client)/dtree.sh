#!/bin/bash
if [ $1 -eq 1 ]; then
   (python3 A3Q1a.py "$2" "$3" "$4" );
elif [ $1 -eq 4 ]; then
   (python3 A3Q1d.py "$2" "$3" "$4" );
elif [ $1 -eq 5 ]; then
   (python3 A3Q1e.py "$2" "$3" "$4" );
elif [ $1 -eq 6 ]; then
   (python3 A3Q1f.py "$2" "$3" "$4" );
elif [ $1 -eq 2 ]; then
   echo "Not done" ;
elif [ $1 -eq 3 ]; then
   echo "Not done" ;
fi
