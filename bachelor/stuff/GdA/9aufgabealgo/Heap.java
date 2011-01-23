public interface Heap {
    // Liefert den Index des linken Kinds des Eintrags an der Position idx.
    // Gibt es keinen solchen Nachfolger, ist der Rueckgabewert -1.
    int left(int idx);
    
    // Liefert den Index des rechten Kinds des Eintrags an der Position idx.
    // Gibt es keinen solchen Nachfolger, ist der Rueckgabewert -1.
    int right(int idx);

    // Liefert den Index des Vaters des Eintrags an der Position idx.
    // Gibt es keinen Vater, ist der Rueckgabewert -1.
    int parent(int idx);
    
    // Fuegt den Wert a unter Erhaltung der Heapeigenschaft in den Heap ein.
    void add(int a);

    // Entfernt das Wurzelelement des Heaps unter Erhaltung der Heapeigenschaft.
    void pop();
    
    // Verallgemeinerung von pop():
    // Entfernt das an der Position idx stehende Element unter Erhaltung der
    // Heapeigenschaft. Hinweis: Gehen Sie zunaechst so vor wie bei pop(), aber
    // beachten Sie Unterschiede bei der Wiederherstellung der Heapstruktur. 
    void del(int idx);

    // Liefert das Element an Position idx oder -1 bei illegalem Index.
    int get(int idx);
    
    // Liefert die Heapgroesse, d.h. die derzeitige Anzahl der verwalteten
    // Elemente.
    int size();
}
