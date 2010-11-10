
public class DFS {
	public void doDFS(Graph G, Node v) {
		
		Keller k = new Keller();
		k.push(v); 	// sozusagen v erstmal in Keller reinhuren...
		v.mark(); 	// v noch als besucht markieren...

		while (!k.isEmpty()) {
			v = k.top(); 	// Node v wird in jedem Schleifendurchlauf
							// zum obersten Element im Keller
			
			k.pop();		// dann oberstes Element, also v entfernen

			Iterator iter = v.getEdges();	// Iterator, der alle Kanten
											// die von v weggehen, enthaelt
			
			while (iter.hastNext()) { 	// solange noch Kanten bestehen...
										// alle auf Keller legen...

				Edge e = (Edge) iter.next();	// e wird zu noch verfuegbarer Kante

				if (e.target.isUnmarked()) {	// Wenn Kantenendknoten noch nicht 
												// markiert, dann ab in den Keller:
					k.push(e.target);
					e.target.mark();	// und markieren, weil wenn in Keller wird es
										// auch besucht.. logisch, ne?
					
				}
			}
		}
	}
}
