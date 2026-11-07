BEGIN {FS=OFS="\t"}
NR==1 {next}
{
  conv=($2=="ConvNetBN"); victim=conv?42:106;
  if ($7=="asd") seconds=conv?36000:108000;
  else if ($7=="telltale") seconds=conv?57600:64800;
  else {
    if (conv) craft=($3==0.001?40:($3==0.002?55:240));
    else craft=($3==0.001?120:($3==0.002?240:464));
    sets=($5==1 && $6==1)?10:30;
    seconds=(30*victim+sets*craft+150)*1.3;
  }
  print "cell_"$1,int((seconds+899)/900)*900,($7=="telltale"?"15G":"6G");
}
