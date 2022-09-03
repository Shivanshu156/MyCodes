// Input 1:
//     A = [4, 5, 2, 10, 8]
// Output 1:
//     G = [-1, 4, -1, 2, 2]
// Explaination 1:
//     index 1: No element less than 4 in left of 4, G[1] = -1
//     index 2: A[1] is only element less than A[2], G[2] = A[1]
//     index 3: No element less than 2 in left of 2, G[3] = -1
//     index 4: A[3] is nearest element which is less than A[4], G[4] = A[3]
//     index 4: A[3] is nearest element which is less than A[5], G[5] = A[3]
    
// Input 2:
//     A = [3, 2, 1]
// Output 2:
//     [-1, -1, -1]
// Explaination 2:
//     index 1: No element less than 3 in left of 3, G[1] = -1
//     index 2: No element less than 2 in left of 2, G[2] = -1
//     index 3: No element less than 1 in left of 1, G[3] = -1







vector<int> Solution::prevSmaller(vector<int> &A) {
    int n = A.size();
    stack<int> st;
    vector<int> G;
    
    for(int i=0;i<n;i++)
    {   
        while(!st.empty())
        {    if (st.top()>=A[i]) st.pop();
            else break;
        }
        if(st.empty()) G[i]=-1;
        else G[i] = st.top();
        cout<<"G["<<i<<"] is "<<G[i]<<endl;
        
        st.push(A[i]);
    }
    
    return A;
    
}
