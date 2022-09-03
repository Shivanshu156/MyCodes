#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <bits/stdc++.h>
#include<queue>
#include<vector>
using namespace std;
// A structure to represent a node in adjacency list
struct AdjListNode
{
    int dest;
    int weight;
    struct AdjListNode* next;
};
 
// A structure to represent an adjacency liat
struct AdjList
{
    struct AdjListNode *head;  // pointer to head node of list
};
 
// A structure to represent a graph. A graph is an array of adjacency lists.
// Size of array will be V (number of vertices in graph)
struct Graph
{
    int V;
    struct AdjList* array;
};
 
// A utility function to create a new adjacency list node
struct AdjListNode* newAdjListNode(int dest, int weight)
{
    struct AdjListNode* newNode =
            (struct AdjListNode*) malloc(sizeof(struct AdjListNode));
    newNode->dest = dest;
    newNode->weight = weight;
    newNode->next = NULL;
    return newNode;
}
 
// A utility function that creates a graph of V vertices
struct Graph* createGraph(int V)
{
    struct Graph* graph = (struct Graph*) malloc(sizeof(struct Graph));
    graph->V = V;
 
    // Create an array of adjacency lists.  Size of array will be V
    graph->array = (struct AdjList*) malloc(V * sizeof(struct AdjList));
 
     // Initialize each adjacency list as empty by making head as NULL
    for (int i = 0; i < V; ++i)
        graph->array[i].head = NULL;
 
    return graph;
}
 
// Adds an edge to an undirected graph
void addEdge(struct Graph* graph, int src, int dest, int weight)
{
    // Add an edge from src to dest.  A new node is added to the adjacency
    // list of src.  The node is added at the begining
    struct AdjListNode* newNode = newAdjListNode(dest, weight);
    newNode->next = graph->array[src].head;
    graph->array[src].head = newNode;
 
    // Since graph is undirected, add an edge from dest to src also
    newNode = newAdjListNode(src, weight);
    newNode->next = graph->array[dest].head;
    graph->array[dest].head = newNode;
}
 
// Structure to represent a min heap node
struct MinHeap
{
    int  v;
    int key;
};

struct Compare
{ bool operator()(const MinHeap & a, const MinHeap & b)
      {return a.key>b.key;
	  }
 } ;

void Display(priority_queue <MinHeap, vector<MinHeap>, Compare> pq)
{ priority_queue <MinHeap, vector<MinHeap>, Compare> g =pq;
while(!g.empty())
{cout<<"("<<g.top().v<<","<<g.top().key<<")\t";
 g.pop();
}
cout<<endl;	
}

bool isinPriorityQueue(priority_queue <MinHeap, vector<MinHeap>, Compare> pq, int v)
{	priority_queue < MinHeap , vector<MinHeap> , Compare > g = pq; 
	bool flag=false;
	while(!g.empty())
	{ if(g.top().v==v)
	   {flag= true; cout<<"Flag is"<<flag<<endl;
	    return true;
	   }
	   g.pop();
	}
	cout<<"Flag is"<<flag<<endl;	
	return false;
}

priority_queue<MinHeap, vector<MinHeap>,Compare>  decreaseKey(priority_queue<MinHeap, vector<MinHeap>,Compare> pq,int v,int key)
 {  priority_queue < MinHeap , vector<MinHeap> , Compare > pq1;
 	while(!pq.empty())
 	{  if(pq.top().v==v)
 	   {  pq.pop();
 	      MinHeap *g = new MinHeap;
 	      g->key=key;
 	      g->v=v;
 	      pq.push(*g);
 	      cout<<"Key Value decreased"<<endl;
		  //pq.top().key = key;Display(pq);
 	    break;}
 	    pq1.push(pq.top());
 	    pq.pop();
	 }
 	while(!pq1.empty())
 	{ pq.push(pq1.top());
 	  pq1.pop();
 	  	
	 }
 	
 	Display(pq);
 	return pq;
 	
 }

 
void printArr(int arr[], int n)
{
    for (int i = 1; i < n; ++i)
        printf("%d - %d\n", arr[i], i);
}
 

void PrimMST(struct Graph* graph)
{
    int V = graph->V;
    int parent[V];   
    int key[V];      
 
    
 	priority_queue<MinHeap, vector<MinHeap>,Compare> pq;
    for (int v = 1; v < V; ++v)
    {   parent[v] = -1;
        key[v] = INT_MAX;
		MinHeap *minHeap = new MinHeap;
		minHeap->v=v; cout<<"MinHeap->v is"<<minHeap->v<<endl;
        minHeap->key=key[v]; cout<<"MinHeap->key is"<<key[v]<<endl;
        pq.push(*minHeap);
       
    }
 
    key[0] = 0;
    MinHeap *rootminHeap = new MinHeap;
		rootminHeap->v=0;
        rootminHeap->key=key[0];
        pq.push(*rootminHeap);
    
 
  cout<<"Size of Priority Queue is "<<pq.size()<<endl;
    
    while (!pq.empty())
    {
        
        MinHeap p = pq.top(); 
		cout<<"pq.top().v = "<<pq.top().v<<endl; cout<<"pq.top().key = "<<pq.top().key<<endl;
        pq.pop();
        Display(pq);
        int u = p.v; 
        cout<<"u is "<<u<<endl;
 
        
        struct AdjListNode* pCrawl = graph->array[u].head;
        while (pCrawl != NULL)
        {
            int v = pCrawl->dest;
 			cout<<"v is "<<v<<endl;
            if (isinPriorityQueue(pq,v) && pCrawl->weight < key[v])
            {
                key[v] = pCrawl->weight;
                parent[v] = u;
                pq = decreaseKey(pq, v, key[v]);
            }
            pCrawl = pCrawl->next;
        }
    }
 
    printArr(parent, V);
}
 

int main()
{

    int V = 9;
    struct Graph* graph = createGraph(V);
    addEdge(graph, 0, 1, 4);
    addEdge(graph, 0, 7, 8);
    addEdge(graph, 1, 2, 8);
    addEdge(graph, 1, 7, 11);
    addEdge(graph, 2, 3, 7);
    addEdge(graph, 2, 8, 2);
    addEdge(graph, 2, 5, 4);
    addEdge(graph, 3, 4, 9);
    addEdge(graph, 3, 5, 14);
    addEdge(graph, 4, 5, 10);
    addEdge(graph, 5, 6, 2);
    addEdge(graph, 6, 7, 1);
    addEdge(graph, 6, 8, 6);
    addEdge(graph, 7, 8, 7);
 
    PrimMST(graph);
 
    return 0;
}
