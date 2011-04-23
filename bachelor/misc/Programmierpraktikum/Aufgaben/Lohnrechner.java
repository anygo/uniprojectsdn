public class Lohnrechner {

	public static void main(String[] args) {
	
	double Bruttolohn_alt = 1767.00;
	double Freibetrag = 200;
	double Bruttolohn = Bruttolohn_alt - Freibetrag;
	double Lohnsteuer = Bruttolohn * 0.16;
	double Krankenversicherung = Bruttolohn * 0.0765;
	double Rentenversicherung = Bruttolohn * 0.0995;
	double Arbeitslosenversicherung = Bruttolohn * 0.0210;
	double Pflegeversicherung = Bruttolohn * 0.0110;
	double Solidaritaetszuschlag = Lohnsteuer * 0.055;
	double Kirchensteuer = Lohnsteuer * 0.08;

	double Abzuege = Lohnsteuer + Krankenversicherung + Rentenversicherung + Arbeitslosenversicherung + Pflegeversicherung + Solidaritaetszuschlag + Kirchensteuer;

	double Prozentsatz = Abzuege / Bruttolohn_alt * 100;

	double Nettolohn = (Bruttolohn - Abzuege);
	System.out.print("\nVon ihrem Bruttolohn von " + Bruttolohn_alt + " Euro bleiben nach den gesetzlichen \nAbzuegen, die insgesamt " + Abzuege + " Euro betragen noch " + Nettolohn + " Euro uerig.\n\n");
	System.out.print("Schicksal!\n" + Prozentsatz + "% fuern Arsch...\n\n\n");


	}

}
