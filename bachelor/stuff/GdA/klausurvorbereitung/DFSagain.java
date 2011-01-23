public void DFS(Graph g, Node v) {
	Keller k = new Keller();
	k.push(v);
	v.mark();
	
	while (!k.isEmpty()) {
		v = k.top();
		k.pop;
		Iterator iter = v.getEdges();
		while (iter.hasNext()) {
			Edge e = (Edge) iter.next();
			if (e.target.isUnmarked()) {
				k.push(e.target);
				e.target.mark();
			}
		}
	}
}
