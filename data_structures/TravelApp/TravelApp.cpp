#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include<iostream>
#include <bits/stdc++.h>
#include<queue>
#include<vector>
 
using namespace std;

struct AdjListNode
{
    long int dest;
    long int Junction_id;
    long int light_time;
    long int build_time;
    long int traverse_time;
    struct AdjListNode* next;
};
 

struct AdjList
{	long int scr;
	long int junction_id;
	long int light_time;
    struct AdjListNode *head;  
};
 
struct Graph
{
    long int V;
    struct AdjList* array;
};

AdjListNode* newAdjListNode(long int dest, long int junction_id,long int light_time, long int build_time, long int traverse_time)
{
    AdjListNode* newNode =new AdjListNode;
    newNode->dest = dest;
    newNode->Junction_id = junction_id;
    newNode->light_time=light_time;
    newNode->build_time=build_time;
    newNode->traverse_time=traverse_time;
    newNode->next = NULL;
    return newNode;
}
 
struct Graph* createGraph(long int V, long int v[][2])
{
    Graph* graph = new Graph;
    graph->V = V;
 
    // Create an array of adjacency lists.  Size of array will be V
    graph->array = new AdjList[1000000];
 
     // Initialize each adjacency list as empty by making head as NULL
    for (long int i = 0; i < V; i++)
	{		graph->array[i].scr=i;
			graph->array[i].junction_id=v[i][0];
			graph->array[i].light_time=v[i][1];
	        graph->array[i].head = NULL;
 
	}

    return graph;
}

Graph *AddJunction(Graph *graph,long int x,long int y) 
{  long int v=graph->V;
	graph->array[v].scr=v;
	graph->array[v].junction_id=x;
	graph->array[v].light_time=y;
	graph->array[v].head = NULL;
	graph->V=graph->V+1;
	//cout<<"done";
	return graph;	
}

// Adds an edge to an undirected graph
Graph *addEdge(Graph* graph, long int edge[])
{
    // Add an edge from src to dest.  A new node is added to the adjacency
    // list of src.  The node is added at the begining
    long int dest=-1,src=-1;
    long int V=graph->V; //cout<<"doubt check V is "<<V<<endl;
	for(long int i=0;i<V;i++)
	if(graph->array[i].junction_id==edge[0])
	{src=i; //cout<<"Source is "<<src<<endl; 
	 break;
	}
	for(long int i=0;i<V;i++)
	if(graph->array[i].junction_id==edge[1])
	{dest=i;// cout<<"Destination is "<<dest; 
	 break;
	}
	if(src==-1 || dest==-1)
	{//cout<<"Unsuccessful edge addition";
		return graph;
	}	
	if(src!=-1 && dest!=-1)
	{
	AdjListNode* newNode = newAdjListNode(dest,graph->array[dest].junction_id,graph->array[dest].light_time,edge[2],edge[3] );
    newNode->next = graph->array[src].head;
    graph->array[src].head = newNode;
//    cout<<"New node added is "<<newNode->Junction_id<<" "<<newNode->light_time<<" "<<newNode->build_time<<" "<<newNode->traverse_time<<endl;
    // Since graph is undirected, add an edge from dest to src also
    newNode = newAdjListNode(src,graph->array[src].junction_id,graph->array[src].light_time,edge[2],edge[3] );
    newNode->next = graph->array[dest].head;
    graph->array[dest].head = newNode;
//    cout<<"New node added is "<<newNode->Junction_id<<" "<<newNode->light_time<<" "<<newNode->build_time<<" "<<newNode->traverse_time<<endl;
	}
	return graph;
	
}
 
void DemolishRoad(Graph *graph,long int x,long int y)
{   int V=graph->V;//cout<<"entres function\n";
	for(long int i=0;i<V;i++)
		{ //cout<<"i = "<<i<<" and current Junction_id is "<<graph->array[i].junction_id<<endl;
				if(graph->array[i].junction_id==x)
			{	//cout<<"In the List of Junction ID "<<graph->array[i].junction_id<<endl;
				AdjListNode *p,*q;
				//cout<<"node found\n";
				p= graph->array[i].head;
			    if(p==NULL) break;
			    if(p->Junction_id==y)
			    {graph->array[i].head=p->next; //cout<<"Demolished\n";
				free(p); break;
				}
				while(p!=NULL)
			    { if(p->Junction_id==y)
			    	{q->next=p->next;
			    	free(p);
					//cout<<"Deleted\n";
					break;
					}
					q=p; p=p->next;
				}
				
			}
			
		}
		
		for(long int i=0;i<V;i++)
		{
			if(graph->array[i].junction_id==y)
			{	AdjListNode *p,*q;
			//	cout<<"node found with junction id \n"<<graph->array[i].junction_id<<endl;
				p= graph->array[i].head;
			//	cout<<"p->Junction Id is "<<p->Junction_id<<endl;
			    if(p==NULL) break;
			   /* if(p->next==NULL && p->Junction_id==x)
			    {graph->array[i].head=NULL; 
				cout<<"Demolished\n";	free(p); return;
				}*/
				if(p->Junction_id==x )
				{graph->array[i].head=p->next;
				 free(p);
				 break;
				}
				while(p!=NULL)
			    { 	//cout<<"p->Junction Id is "<<p->Junction_id<<endl;
					if(p->Junction_id==x)
			    	{q->next=p->next;
			    	free(p);
					//cout<<"Demolished\n";
					break;
					}
					q=p; p=p->next;
				}
				
			}
			
		}
	
 } 
 
struct MinHeap
{
    long int  v;
    long int key;
};
 
struct Compare
{ bool operator()(const MinHeap & a, const MinHeap & b)
      {return a.key>b.key;
	  }
 } ;
 
 void Display(priority_queue <MinHeap, vector<MinHeap>, Compare> pq)
{ priority_queue <MinHeap, vector<MinHeap>, Compare> g =pq;
while(!g.empty())
{//cout<<"("<<g.top().v<<","<<g.top().key<<")\t";
 g.pop();
}
//cout<<endl;	
}
 
bool isinPriorityQueue(priority_queue <MinHeap, vector<MinHeap>, Compare> pq,long  int v)
{	priority_queue < MinHeap , vector<MinHeap> , Compare > g = pq; 
	bool flag=false;
	while(!g.empty())
	{ if(g.top().v==v)
	   {flag= true; //cout<<"Flag is"<<flag<<endl;
	    return true;
	   }
	   g.pop();
	}
//	cout<<"Flag is"<<flag<<endl;	
	return false;
}

priority_queue<MinHeap, vector<MinHeap>,Compare>  decreaseKey(priority_queue<MinHeap, vector<MinHeap>,Compare> pq,long int v,long int key)
 {  priority_queue < MinHeap , vector<MinHeap> , Compare > pq1;
 	while(!pq.empty())
 	{  if(pq.top().v==v)
 	   {  pq.pop();
 	      MinHeap *g = new MinHeap;
 	      g->key=key;
 	      g->v=v;
 	      pq.push(*g);
 	      //cout<<"Key Value decreased"<<endl;
		  //pq.top().key = key;Display(pq);
 	    break;}
 	    pq1.push(pq.top());
 	    pq.pop();
	 }
 	while(!pq1.empty())
 	{ pq.push(pq1.top());
 	  pq1.pop();
 	  	
	 }
 	
 //	Display(pq);
 	return pq;
 	
 }
 

void MergeSort(long int a[][2],  long int l,  long int r)
{
	if (l < r)
    {   long int m = (r+l)/2;
        MergeSort(a, l, m);
        MergeSort(a, m+1, r);
		//////Merging Starts
					   long int n1=m-l+1, n2=r-m;
					 	long int L[n1][2], R[n2][2];
					 
					    for(  long int i=0;i<n1;i++)
						{  L[i][0] = a[l + i][0];
						   L[i][1] = a[l + i][1];
						}   
						for ( long int j = 0; j < n2; j++)
					    { R[j][0] = a[m+1+j][0];
					      R[j][1] = a[m+1+j][1];
						}    
					 
					    long int i=0,j=0,k=l; 
					    while (i < n1 && j < n2)
					    {
					        if (L[i][0] < R[j][0])
					        {	a[k][0] = L[i][0]; 
								a[k][1] = L[i][1];
					            i++;
					        }
					        else if (L[i][0] > R[j][0])
					        {	a[k][0] = R[j][0];
					            a[k][1] = R[j][1];
					            j++;
					        }
					        else if (L[i][0] == R[j][0])
					        if(L[i][1] <= R[j][1])
					        {	a[k][0] = L[i][0]; 
								a[k][1] = L[i][1];
					            i++;
					        }
					        else{	a[k][0] = R[j][0];
					            a[k][1] = R[j][1];
					            j++;
					        }
					        
					        k++;
					    }
					    while (i < n1)
					    {
					        a[k][0] = L[i][0]; a[k][1] = L[i][1];
					        i++;
					        k++;
					    }
					    while (j < n2)
					    {	a[k][1] = R[j][1];
					        a[k][0] = R[j][0];
					        j++; k++;
					    }
					
		//////Merging Ends
	}
}


void printArr(long int arr[], long int n, Graph *G)
{   //cout<<"Parent array is ";
	//for(int i=0;i<n;i++)
	//cout<<arr[i]<<"  ";
	//cout<<endl<<"no of edges is "<<n-1<<"\n"; 
    for(long int i=1;i<n;i++)
        if(arr[i]==-1)
        {cout<<"-1"<<endl; return;}
	cout<<n-1<<" ";
	long int weight=0;
    for (long int i = 1; i < n; ++i)
    {   //cout<<"Value of i is "<<i;
		long int V=G->V; long int j=arr[i];
		AdjListNode *r =G->array[j].head;
    	while(r!=NULL)
    	{
    		if(r->Junction_id==G->array[i].junction_id)
    		{	//cout<<"weight = "<<weight;
				weight=weight+r->build_time;
    		    //cout<<"+ "<<r->build_time<<" = "<<weight<<endl;
			 break;
			}
    		r=r->next;
    		
		//	cout<<"Value of V is "<<V<<" in while loop"<<endl;
		}
    //	cout<<"IN for loop"<<endl;
		}	
    cout<<weight<<" ";
    long int a[n-1][2];
	for (long int i = 1; i < n; ++i)
    if(G->array[arr[i]].junction_id <G->array[i].junction_id)
     { a[i-1][0]=G->array[arr[i]].junction_id; 
	   a[i-1][1]=G->array[i].junction_id;}
    else 
     {a[i-1][1]=G->array[arr[i]].junction_id;a[i-1][0]=G->array[i].junction_id;
	   }  	
   	MergeSort(a,0,n-2);
	   for(long int i=0;i<n-1;i++)
   	cout<<a[i][0]<<" "<<a[i][1]<<" ";
     	
    cout<<endl;	

}


void printArrdijkstra(Graph *graph,long int src, long int des,long int dist[],long int parent[], long int n)
{   
    if(dist[des]==LONG_MAX)
    {cout<<"-1"<<endl; return;}
   // printf("Vertex   Distance from Source\n");
   //cout<<"parent array is ";
   //for(int i=0;i<n;i++)
   //cout<<graph->array[i].junction_id<<"-"<<graph->array[parent[i]].junction_id<<" ";
   long int k=0; long int j_id[n];
   cout<<dist[des]<<" ";
   long int i=des;
   while(parent[i]!=-1)
   {   j_id[k]=graph->array[i].junction_id;
   	i=parent[i];
   	    k++;
   }
   k--;
   cout<<k+2<<" ";
   cout<<graph->array[src].junction_id<<" ";
   for(long int i=k;i>=0;i--)
   cout<<j_id[i]<<" ";
   
   cout<<endl;
   
		//printf("%d \t\t %d\n", i, dist[i]);
}
 

void PrimMST(struct Graph* graph)
{
    long int V = graph->V;// Get the number of vertices in graph
    long int parent[V];   // Array to store constructed MST
    long int key[V];      // Key values used to pick minimum weight edge in cut
    if(V<=1)
    {cout<<"-1"<<endl; return;}
    priority_queue<MinHeap, vector<MinHeap>,Compare> pq;
    for (long int v = 1; v < V; ++v)
    {
        parent[v] = -1;
        key[v] = INT_MAX;
        MinHeap *minHeap = new MinHeap;
		minHeap->v=v; //cout<<"MinHeap->v is"<<minHeap->v<<endl;
        minHeap->key=key[v]; //cout<<"MinHeap->key is"<<key[v]<<endl;
        pq.push(*minHeap);
    }
 
    
    key[0] = 0;
    MinHeap *rootminHeap = new MinHeap;
		rootminHeap->v=0;
        rootminHeap->key=key[0];
        pq.push(*rootminHeap);
 
 //    cout<<"Size of Priority Queue is "<<pq.size()<<endl;
 
   
    while (!pq.empty())
    {
        MinHeap p = pq.top(); 
        //cout<<"pq.top().v = "<<pq.top().v<<endl; cout<<"pq.top().key = "<<pq.top().key<<endl;
        pq.pop();
        Display(pq);
        long int u = p.v; 
       // cout<<"u is "<<u<<endl;
        
		struct AdjListNode* pCrawl = graph->array[u].head;
        while (pCrawl != NULL)
        {
            long int v = pCrawl->dest;
 
           // cout<<"v is "<<v<<endl;
            if (isinPriorityQueue(pq,v) && pCrawl->build_time < key[v])
            {
                key[v] = pCrawl->build_time;
                parent[v] = u;
                pq = decreaseKey(pq, v, key[v]);
            }
            pCrawl = pCrawl->next;
        }
    }
 
    // print edges of MST
    printArr(parent, V, graph);
}
 


void dijkstra(struct Graph* graph,long int j_id1,long int j_id2)

{   
    long int V = graph->V;
    long int dist[V],parent[V];
 	long int src,des;
 	for(long int i=0;i<V;i++)
 	if(graph->array[i].junction_id==j_id1)
 	src=i; 
 	//cout<<"Source index is "<<src<<" and Source J_id is "<<graph->array[src].junction_id<<endl;
 	for(long int i=0;i<V;i++)
 	if(graph->array[i].junction_id==j_id2)
 	des=i;
 	//cout<<"Destination index is "<<des<<" and Destination J_id is "<<graph->array[des].junction_id<<endl;
	for(long int v=0;v<V;v++)
	parent[v]=-1;
   priority_queue<MinHeap, vector<MinHeap>,Compare> pq;
   // struct MinHeap* minHeap = createMinHeap(V);

	 for (long int v = 0; v < V; ++v)
    {
        dist[v] = LONG_MAX;
        MinHeap *minHeap = new MinHeap;
		minHeap->v=v; //cout<<"MinHeap->v is"<<minHeap->v<<endl;
        minHeap->key=dist[v]; //cout<<"MinHeap->key is"<<key[v]<<endl;
        pq.push(*minHeap);
       
    }
 
   dist[src] = 0;
   MinHeap *rootminHeap = new MinHeap;
		rootminHeap->v=src;
        rootminHeap->key=dist[src];
        pq.push(*rootminHeap);
    
    
    decreaseKey(pq, src, dist[src]);
 
 

    while (!pq.empty())
    {
        
        MinHeap p = pq.top(); 
		//cout<<"pq.top().v = "<<pq.top().v<<endl; cout<<"pq.top().key = "<<pq.top().key<<endl;
        pq.pop();
        //Display(pq);
       long  int u = p.v; 
        //cout<<"u is "<<u<<endl;
        

        
        
        struct AdjListNode* pCrawl = graph->array[u].head;
       // cout<<"Junction Id of u is "<<graph->array[u].junction_id<<endl;
        while (pCrawl != NULL)
        {
            long int v = pCrawl->dest;
         //   cout<<"Checking for node with Junction_ID "<<graph->array[u].head->Junction_id<<" traverse time "<<graph->array[u].head->traverse_time<<" and build time "<<graph->array[u].head->build_time<<endl;
 			long int light; //cout<<"before light\n";
           //   cout<<"(dist[u]("<<dist[u]<<")+pCrawl->traverse_time("<<pCrawl->traverse_time<<")%pCrawl->light_time("<<pCrawl->light_time<<") = "<<(dist[u]+pCrawl->traverse_time)%pCrawl->light_time<<endl;
			if((dist[u]+pCrawl->traverse_time)%pCrawl->light_time !=0 && v!=des )
             light=pCrawl->light_time - (dist[u]+pCrawl->traverse_time)%pCrawl->light_time;
             else light =0;
         //   cout<<" Additional light time is "<<light<<endl;
            if (isinPriorityQueue(pq,v) && dist[u] != INT_MAX && 
                                          (pCrawl->traverse_time +light+ dist[u] )< dist[v])
            {
                dist[v] = dist[u] + light+ pCrawl->traverse_time;
 		//		cout<<"Updated Distance is "<<dist[u]<<" + "<< pCrawl->traverse_time<<" + "<<light<<" = "<<dist[v]<<endl;
                                pq = decreaseKey(pq, v, dist[v]);
                                parent[v]=u;
            }
            pCrawl = pCrawl->next;
        }
    }
 
    
    printArrdijkstra(graph,src,des,dist,parent, V);
     //cout<<"\ntraverse time is "<<dist[des];

     //cout<<endl;
}



int main()
{
    long int V,E,Q,fun;
    cin>>V>>E;
    long int vertex[V][2],edge[E][4];
    for(long int i=0;i<V;i++)
    cin>>vertex[i][0]>>vertex[i][1];
    for(long int i=0;i<E;i++)
    cin>>edge[i][0]>>edge[i][1]>>edge[i][2]>>edge[i][3];
    Graph* graph = createGraph(V,vertex);
    for(long int i=0;i<E;i++)
    {long int Edge[4];
    	for(long int j=0;j<4;j++)
    	Edge[j]=edge[i][j];
    	graph=addEdge(graph,Edge);
    }
    cin>>Q;
	for(long int i=0;i<Q;i++)
    { 	//cout<<i+1<<". ";
	    cin>>fun;
        
    	switch(fun)
		{ 
		  
		  case 1:  long int x,y; cin>>x>>y; graph=AddJunction(graph,x,y); break;
		  case 2:  long int edg[4]; cin>>edg[0]>>edg[1]>>edg[2]>>edg[3]; graph= addEdge(graph,edg); break;  							
		  case 3:  long int a,b; cin>>a>>b; DemolishRoad(graph,a,b); break;
		  case 4:  PrimMST(graph); break;
		  case 5:  long int pa,pb; cin>>pa>>pb;	dijkstra(graph,pa,pb);						break;
		  default: break;	
		           
		}
	}
    
 
    
}
