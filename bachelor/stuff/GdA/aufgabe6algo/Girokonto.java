public class Girokonto extends Konto
{
    protected double dispo;
   
    public Girokonto(String name, double dispolimit)
    {
        super(name);
    }
   
    public boolean deckungPruefen(double betrag)
    {
        if(saldo + dispo - betrag < 0)
            return false;
        else
            return true;
    }

    public void ueberweisen(Konto empfaenger, double betrag)
    {
        if(betrag < 0) {
            System.out.println("Betrag muss groesser 0 sein.");
            return;
        }
        
        if(deckungPruefen(betrag) == false) {
            System.out.println("Konto nicht ausreichend gedeckt.");
            return;
        }
         
        abheben(betrag);
        empfaenger.einzahlen(betrag);
    }

    public double berechneZins()
    {
        return 0;
    }
}

