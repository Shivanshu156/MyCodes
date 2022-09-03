// Input 1:
//     A = "{A:"B",C:{D:"E",F:{G:"H",I:"J"}}}"
// Output 1:
//     { 
//         A:"B",
//         C: 
//         { 
//             D:"E",
//             F: 
//             { 
//                 G:"H",
//                 I:"J"
//             } 
//         } 
//     }


vector<string> Solution::prettyJSON(string A) {
    int n = A.size(), tab_count=0;
    string word; vector<string> B;
    for(int i=0;i<n;i++)
    {   
        if((A[i]=='{' || A[i]=='[' || A[i]==']' || A[i]=='}') && A[i+1] == ',')
        {   if(word.size()) {B.push_back(word); word.clear();}
            word.push_back(A[i]);
            word.push_back(A[i+1]);
            B.push_back(word);
            word.clear();
            i++;
        }
        else if(A[i]=='{' || A[i]=='[' || A[i]==']' || A[i]=='}')
        {   if(word.size()) {B.push_back(word); word.clear();}
            word.push_back(A[i]);
            B.push_back(word);
            word.clear();
        }
        else if(A[i]==',')
        {   word.push_back(A[i]);
            B.push_back(word);
            word.clear();
        }
        else if(A[i]!=' ') word.push_back(A[i]);
    }
    
    
    for(int i=0;i<B.size();i++)
    {
        // cout<<"tab count is "<<tab_count<<endl;
        if(B[i][B[i].size()-1]==']' || B[i][B[i].size()-1] == '}' || B[i][B[i].size()-2]==']' || B[i][B[i].size()-2] == '}') tab_count--;
        for(int k=0;k<tab_count;k++)
        B[i] = '\t' + B[i];
        // cout<<"B[i] is "<<B[i]<<endl;
        if(B[i][B[i].size()-1]=='{' || B[i][B[i].size()-1] == '[') tab_count+=1;
        
    }
    // cout<<"hello" + B[4] + "hello"<<endl;
    return B;
}
