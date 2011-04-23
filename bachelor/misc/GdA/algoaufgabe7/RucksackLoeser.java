public interface RucksackLoeser {
    final int NICHT_PACKEN = 0;
    final int PACKEN = 1;
    final int UNMOEGLICH = -1;

    void solve();

    void printmatrix();
    void printsolution();
}
