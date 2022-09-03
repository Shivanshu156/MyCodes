//  Example:
// If array A is [4, 1, 1, 2, 1, 3]
// and value elem is 1, 
// then new length is 3, and A is now [4, 2, 3] 


int Solution::removeElement(vector<int> &A, int B) {
    int i=0, count=0;
    
    while(i<A.size())
    {
     if(A[i]==B) break;   
        i++;count++;
    }    
    int j=i+1;
    while(j<A.size())
    {
        if(A[j]!=B)
        {
            A[i]=A[j]; i++; count++;
        }
        j++;
        
    }
    return count;
    
    
}