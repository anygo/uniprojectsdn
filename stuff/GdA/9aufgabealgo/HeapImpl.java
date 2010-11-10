
public class HeapImpl implements Heap {
	
	private int heapsize;
	private boolean heap_ist_voll;
	private int[] heap;

	// Konstruktor
	public HeapImpl(int capacity) {
		heap = new int[capacity];
		heapsize = 0;
		if (capacity == 0) {
			heap_ist_voll = true;
			System.out.println("Ein Heap der Groesse 0 macht wenig Sinn ;)");
		} else {
		heap_ist_voll = false;
		}
	}
	
	public int left(int idx) {
		return (2*idx + 1);
	}
	
	public int right(int idx) {
		return (2*idx + 2);
	}

	public int parent(int idx) {
		return ((idx - 1) / 2);
	}

	public void add(int a) {
		if (heap_ist_voll == true) {
			System.out.println("Fehler: Kann Element ("+a+") nicht hinzufuegen. Heap ist voll.");
			return;
		}
		
		heapsize++;
		if (heapsize == heap.length) {
			heap_ist_voll = true;
		}	

		int tmp_idx = heapsize - 1;
		heap[tmp_idx] = a;

		while (a > heap[parent(tmp_idx)]) {
		heap[tmp_idx] = heap[parent(tmp_idx)]; 	
		heap[parent(tmp_idx)] = a;
		tmp_idx = parent(tmp_idx);
		}	
	}

	public void pop() {
		if (size() == 0) {
			System.out.println("Fehler: Heap besitzt kein Wurzelelement");
			return;
		}
		
		// Letzes Element mit Wurzel vertauschen...
		heap[0] = heap[size() - 1];
		heapsize--;

		int tmp_idx = 0;

		while (right(tmp_idx) <= size()) {
			if (heap[tmp_idx] < heap[left(tmp_idx)] || heap[tmp_idx] < heap[right(tmp_idx)]) {
				int tmp = heap[tmp_idx];
				if (heap[left(tmp_idx)] <= heap[right(tmp_idx)]) {
					heap[tmp_idx] = heap[right(tmp_idx)];
					heap[right(tmp_idx)] = tmp;
					tmp_idx = right(tmp_idx);
				} else {
					heap[tmp_idx] = heap[left(tmp_idx)];
					heap[left(tmp_idx)] = tmp;
					tmp_idx = left(tmp_idx);
				}
			} else break;
		}
	}

	public void del(int idx) {
		if (idx > size() - 1 || idx < 0) {
			System.out.println("Fehler: Index "+idx+" nicht belegt.");
			return;
		}
		
		// Zu loeschendes Element mit letztem Element vertauschen...
		heap[idx] = heap[size() - 1];
		heapsize--;

		if (heap[idx] > heap[parent(idx)]) {
			while (heap[idx] > heap[parent(idx)]) {
				int tmp = heap[parent(idx)];
				heap[parent(idx)] = heap[idx];
				heap[idx] = tmp;
			}
		} else {
			
			while (right(idx) <= size()) {
				
				if (heap[idx] < heap[left(idx)] || heap[idx] < heap[right(idx)]) {
					
					int tmp = heap[idx];
					if (heap[left(idx)] <= heap[right(idx)]) {
						heap[idx] = heap[right(idx)];
						heap[right(idx)] = tmp;
						idx = right(idx);
					} else {
						heap[idx] = heap[left(idx)];
						heap[left(idx)] = tmp;
						idx = left(idx);
					}

				} else break;
			}

		}
	}

	public int get(int idx) {
		if (idx < 0) {
			System.out.println("Fehler: Index < 0 nicht zulaessig");
			return -1;
		}
		if (idx >= size()) {
			System.out.println("Fehler: Kein Element an Index "+idx+".");
			return -1;
		}
		return (heap[idx]);
	}

	public int size() {
		return heapsize;
	}

	public String toString() {
		String blaaa = new String();
		for (int i = 0; i < size(); i++) {
			blaaa += heap[i] + " ";
		}
		return blaaa;
	}

	public static void main(String[] args) {
		HeapImpl test = new HeapImpl(500);
		for (int i = 0; i < 480; i++) {
			test.add(i);
		}
		for (int i = 235; i > 0; i--) {
			test.del(i);
			test.pop();
		}
		System.out.println("\n"+test);
	}

}
