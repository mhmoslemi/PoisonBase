function seconds(t,a,n,d) {n=split(t,a,/-/); d=(n==2?a[1]:0); t=a[n]; n=split(t,a,/:/); return d*86400+(n==2?a[1]*60+a[2]:a[1]*3600+a[2]*60+a[3])}
NR==FNR {
    split($0,a,"\t"); n=split(a[5],p,",");
    for(i=1;i<=n;i++) {id=a[2]; if(a[1]!~/^(aggregate|wo_)/)id=id"_"p[i]; mem[id]=a[6]; time[id]=seconds(a[4]); deps[id]=a[3]; expected++}
    next
}
{
    split($0,a,"|"); id=a[1]; actualdep=a[8]; gsub(/\([^)]*\)/,"",actualdep);
    gsub(/_[*]/,"",actualdep); gsub(/,after(ok|any):/,":",actualdep);
    if(!(id in mem) || seen[id]++ || a[2]!="def-bghaddar_gpu" || a[3]!="PENDING" || a[4]!=mem[id] || seconds(a[5])!=time[id] || a[6]!=1 || a[7]!~/h100/ || actualdep!=deps[id]) {print "MISMATCH",$0; bad++}
    actual++; count[a[4]]++
}
END {print "Expected",expected,"observed",actual,"mismatches",bad+0; for(m in count)print "RAM",m,count[m]; if(actual!=expected || bad)exit 1}
