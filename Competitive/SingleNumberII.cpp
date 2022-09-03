int Solution::singleNumber(const vector<int> &A) {
    int bit[32]={0}, ans=0;    
    for(int i=0;i<32;i++)
    {   for(int j=0; j<A.size();j++)
        if((A[j] & (1<<i))!=0)
        bit[i]++;
        bit[i]%=3;
        // cout<<"Bit["<<i<<"] = "<<bit[i];
        ans = ans|(bit[i]<<i);
    }
    // for(int i=0;i<32;i++)
    // cout<<bit[i]<<" ";
    return ans;
    }
