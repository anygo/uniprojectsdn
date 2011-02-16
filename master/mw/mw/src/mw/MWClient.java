package mw;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.StringTokenizer;

import javax.xml.namespace.QName;

import mw.facebook.imported.*;


public class MWClient {
	
	private MWMyFacebookService facebook; 
	public long lastOperationDelay;

	public MWClient()
	{
		if (System.getProperty("MW_SERVER_URL") == null) {
			System.err.println("MW_SERVER_URL not set!");
			System.exit(-1);
		}
		URL wsdlLocation;
		try {
			wsdlLocation = new URL(System.getProperty("MW_SERVER_URL"));
			MWFacebookService service = new MWFacebookService(wsdlLocation, new QName("http://facebook.mw/", "MWFacebookService"));
			facebook = service.getMWMyFacebookServicePort();
		} catch (MalformedURLException e) {
			e.printStackTrace();
		}
		
		lastOperationDelay = 0;
	}
	
	// Frage Server alle i Sekunden nach Auslastung
	public int status(int i) { 
		
		return facebook.getServerStatus(i);
	}
	
	public void executor(int poolsize, int ntimes, String query) {
		
		new MWClientThreadPool(this, query, poolsize, ntimes);
	}

	public void searchIDs(String name)
	{
		long start = System.currentTimeMillis();
		mw.facebook.imported.StringArray sa = facebook.searchIDs(name);
		lastOperationDelay = System.currentTimeMillis() - start; 

		for (String s: sa.getItem())
		{
			//try {
			//	System.out.println("Name: " + facebook.getName(s) + " - ID: " + s);
			//	System.out.println("ID: " + s);
			//} catch (MWUnknownIDException_Exception e) {
			//	e.printStackTrace();
			//}
		}
	}
	
	public void getFriends(String id)
	{
		try {
			long start = System.currentTimeMillis();
			mw.facebook.imported.StringArray sa = facebook.getFriends(id);
			lastOperationDelay = System.currentTimeMillis() - start; 
			
			System.out.println(sa.getItem().size() + " " + "Freund(e)");
			for (String s: sa.getItem())
			{
				System.out.println("Name: " + facebook.getName(s) + " - ID: " + s);
			}
		} catch (MWUnknownIDException_Exception e) {
			System.err.println("Fehler bei Aufruf von getFriends(). ID ungueltig?");
			System.err.println(e.getMessage());
		}
	}
	
	public static void main(String[] args) throws IOException 
	{
		MWClient client = new MWClient();
		
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		
		
		
		while (true) {
			System.out.println("'S' fuer searchIDs ODER 'G' fuer getFriends ODER 'P' (oder T) fuer CalculatePath eingeben.");
			String str = br.readLine();
			if (str.equalsIgnoreCase("S")) {
				System.out.println("Namen eingeben: ");
				str = br.readLine();
				client.searchIDs(str);
				System.out.println("searchIDs(" + str + ") took " + client.lastOperationDelay + " ms");
			}
			else if (str.equalsIgnoreCase("G")) {
				System.out.println("ID eingeben: ");
				str = br.readLine();
				client.getFriends(str);
				System.out.println("getFriends(" + str + ") took " + client.lastOperationDelay + " ms");
			}
			else if (str.equalsIgnoreCase("STATUS")) {
				System.out.println(client.status(Integer.parseInt(br.readLine())));
			}
			else if (str.equalsIgnoreCase("executor")) {
				str = br.readLine();
				StringTokenizer st = new StringTokenizer(str, " ");
				client.executor(Integer.parseInt(st.nextToken()), Integer.parseInt(st.nextToken()), st.nextToken());
			}
			else {
				System.err.println("falsche Eingabe.");
			}
			System.out.println();
		}
	}
}
