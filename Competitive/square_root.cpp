int Solution::sqrt(int A) {
    int start =1, mid, end=A;
    if(A==0) return 0;
    while(start<=end)
    {   mid = (start+end)/2;
        if(mid == A/mid)
        return mid;
        if(mid>A/mid)
        end = mid-1;
        if(mid<A/mid)
        if(mid+1>A/(mid+1)) return mid;
        else    start = mid+1;
    }
    
    
    
    
}