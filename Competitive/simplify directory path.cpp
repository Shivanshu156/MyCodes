// "/a/./"   --> means stay at the current directory 'a'
// "/a/b/.." --> means jump to the parent directory
//               from 'b' to 'a'
// "////"    --> consecutive multiple '/' are a  valid  
//               path, they are equivalent to single "/".

// Input : /home/
// Output : /home

// Input : /a/./b/../../c/
// Output : /c

// Input : /a/..
// Output:/

// Input : /a/../
// Output : /

// Input : /../../../../../a
// Output : /a

// Input : /a/./b/./c/./d/
// Output : /a/b/c/d

// Input : /a/../.././../../.
// Output:/

// Input : /a//b//c//////d
// Output : /a/b/c/d



string tostring(char x)
{
    string s(1,x);
    return s;

}
string Solution::simplifyPath(string A) {
    int n = A.size(); string ans ="";
    
    stack<string> st;
    for(int i=0;i<n;i++)
    if(st.empty() && A[i]!='/') {   st.push("/"); st.push(tostring(A[i]));}    
    else if(st.empty()) st.push(tostring(A[i]));
    else if(A[i]!= '/' && A[i]!= '.'){    st.push(tostring(A[i])); }
    else if(A[i]=='/') if(st.top()=="/") continue; 
                      else if(st.top()==".") while(st.top()!="/") st.pop(); 
                      else st.push(tostring(A[i]));
    else if(A[i]=='.') if(st.top()==".") {  while(st.top()!="/") st.pop(); st.pop();
                                            if(!st.empty()){while(st.top()!="/") st.pop(); st.pop();} 
                                            else continue;}
                      else st.push(tostring(A[i]));
    
    
    if(st.empty()) return "/";
    else if(st.top() == "/") st.pop();    
    else if(st.top()==".") {while(st.top()!="/") st.pop(); st.pop();}
    if(st.empty()) return "/"; 
    while(!st.empty()){
         ans += st.top(); st.pop();}
    // return ans;
    reverse(ans.begin(), ans.end());
    return ans;
    
    
}