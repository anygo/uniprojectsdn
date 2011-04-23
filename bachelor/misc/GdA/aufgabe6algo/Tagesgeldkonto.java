public class Tagesgeldkonto extends Konto
{
    protected static double zinssatz = 3.8;

    public Tagesgeldkonto(String name)
    {
        super(name);
    }

    public static void setzeZinssatz(double zs)
    {
        if(zinssatz < 0)
            System.out.println("Zinssatz muss groesser 0 sein.");
        zinssatz = zs;
    }

    public void zinsgutschrift()
    {
        double zins = zinssatz * saldo / 100;
        System.out.println("Zinsgutschrift: " + zins);
        saldo += zins;
        System.out.println("  Neuer Saldo von " + inhaber + ": " + saldo);
    }
}
