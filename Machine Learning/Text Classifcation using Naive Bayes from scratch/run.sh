#!/bin/bash
if [ $1 -eq 1 ]; then
   if [ "$4" == "a" ] || [ "$4" == "b" ] || [ "$4" == "c" ]; then
      (python3 A2Q1.py "$2" "$3" "$4" ) ;
   elif [ "$4" == "d" ]; then
      python3 A2Q1_d.py "$2" "$3" "$4" ;
   elif [ "$4" == "e" ] || [ "$4" == "f" ]; then
      python3 A2Q1_e.py "$2" "$3" "$4" ;
   elif [ "$4" == "g" ] ; then
      python3 A2Q1_g.py "$2" "$3" "$4" ;
   fi
elif [ $1 -eq 2 ]; then
   echo "Not done" ;
fi
