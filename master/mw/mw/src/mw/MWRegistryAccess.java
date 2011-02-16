package mw;

import java.net.PasswordAuthentication;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.Properties;
import java.util.Set;

import javax.xml.registry.*;
import javax.xml.registry.infomodel.InternationalString;
import javax.xml.registry.infomodel.Organization;
import javax.xml.registry.infomodel.Service;
import javax.xml.registry.infomodel.ServiceBinding;


public class MWRegistryAccess {
	
	private Connection connection;
	private RegistryService regSvc;
	private BusinessLifeCycleManager lcm;
	
	
	// Authentifizierung
	public void authenticate(String userName, String password)
	{	
		PasswordAuthentication pa = new PasswordAuthentication(userName, password.toCharArray());
		Set<PasswordAuthentication> credentials = new HashSet<PasswordAuthentication>();
		credentials.add(pa);
		try {
			connection.setCredentials(credentials);
		} catch (JAXRException e) {
			System.err.println("Fehler beim Setzen der Credentials:" + e.getMessage());
		}
		
	}
	
	// Service registrieren
	@SuppressWarnings("unchecked")
	public void registerService(String orgName, String serviceName, String wsdlURL)
	{	
		Organization org = null;
		
		Collection<String> findQualifiers = new ArrayList<String>();
		findQualifiers.add(FindQualifier.EXACT_NAME_MATCH);
		
		Collection<String> namePatterns = new ArrayList<String>();
		namePatterns.add(orgName);

		BusinessQueryManager m;
		@SuppressWarnings("rawtypes")
		Collection exceptions = null;
		try 
		{
			m = regSvc.getBusinessQueryManager();
		} catch (JAXRException e1) {
			System.err.println("Problem mit BusinessQueryManager.");
			e1.printStackTrace();
			return;
		}		

		BulkResponse br;
		Collection<Organization> orgs;
		try {
			br = m.findOrganizations(findQualifiers, namePatterns, null, null, null, null);
			orgs = br.getCollection();

			
			if(orgs.size() > 0) 
			{
				for (Organization o: orgs)
				{
					System.out.println("Loesche Service von Organisation " + o.getName().getValue());
					o.removeServices(o.getServices());
					org = o;
				}	
			}
			else
			{
				InternationalString onis = lcm.createInternationalString(orgName);
				org = lcm.createOrganization(onis);		
			}
		} catch (JAXRException e1) {
			System.err.println(e1.getMessage());
		}


		try {
			InternationalString snis = lcm.createInternationalString(serviceName);
			Service svc = lcm.createService(snis);
			org.addService(svc);
			
			ServiceBinding binding = lcm.createServiceBinding();
			binding.setAccessURI(wsdlURL);
			svc.addServiceBinding(binding);
			
			
			orgs = new ArrayList<Organization>(1);
			orgs.add(org);
			br = lcm.saveOrganizations(orgs);
			
			exceptions = br.getExceptions();
		} catch (JAXRException e) {
			System.err.println("Fehler bei registerService: " + e.getMessage());
			return;
		}
		
		if (exceptions != null) {
		    System.err.println(exceptions.size() + " Exception(s) beim :-(");
		} 
	}

	// Verbindungsaufbau
	public void openConnection(String queryManagerURL, String lifeCycleManagerURL)
	{
		Properties props = new Properties();
		props.setProperty("javax.xml.registry.queryManagerURL", queryManagerURL);
		props.setProperty("javax.xml.registry.lifeCycleManagerURL", lifeCycleManagerURL);
	
		try {
			ConnectionFactory fact = ConnectionFactory.newInstance();
			fact.setProperties(props);
			connection = fact.createConnection();
			regSvc = connection.getRegistryService();
			lcm = regSvc.getBusinessLifeCycleManager();
			
		} catch (JAXRException e) {
			System.err.println("Es konnte keine Verbindung hergestellt werden. Exit.");
			e.printStackTrace();
			System.exit(-1);
		}
	}
	
	// Verbindungsabbau
	public void closeConnection()
	{
		try {
			connection.close();
		} catch (JAXRException e) {
			System.err.println("Fehler bei connection.close().");
			e.printStackTrace();
		}
	}
	
	// WSDLs ausgeben
	@SuppressWarnings("unchecked")
	public void listWSDLs(String serviceName)
	{
		System.out.println("Suche nach: " + serviceName);
		Collection<String> findQualifiers = new ArrayList<String>();
		findQualifiers.add(FindQualifier.SORT_BY_NAME_ASC);
		
		Collection<String> namePatterns = new ArrayList<String>();
		namePatterns.add(serviceName);

		
		BusinessQueryManager m;
		try {
			m = regSvc.getBusinessQueryManager();
		} catch (JAXRException e1) {
			System.err.println("Problem mit BusinessQueryManager.");
			e1.printStackTrace();
			return;
		}		
		
		try {
			BulkResponse br = m.findServices(null, findQualifiers, namePatterns, null, null);
			Collection<Service> services = br.getCollection();
			
			
			
			// Ausgabe
			for (Service s: services)
			{
				Collection<ServiceBinding> serviceBindings = s.getServiceBindings();
				System.out.println("------------------------------");
				for (ServiceBinding sb: serviceBindings)
				{
					System.out.println("Name: " + s.getName().getValue());
					System.out.println("WSDL: " + sb.getAccessURI());
					System.out.println("Organisation: " + s.getProvidingOrganization().getName().getValue());
					System.out.println("------------------------------");
				}
			}		
		} catch (JAXRException e) {
			System.err.println("Fehler bei Kommunikation mit Registry Service.");
			e.printStackTrace();
			return;
		}	
			
	}
	
	

	public static void main(String[] args) 
	{
		if (args.length < 1)
		{
			System.err.println("Bitte Programm richtig aufrufen! (");
			return;
		}
		
		// Infos
		String registryURL = "http://faui48.informatik.uni-erlangen.de:18080/juddi";
		String queryManagerURL = registryURL + "/inquiry";
		String lifeCycleManagerURL = registryURL + "/publish";
		
		System.out.println("Starte Verbindungsaufbau zu " + registryURL + "...");
		MWRegistryAccess registry = new MWRegistryAccess();
		registry.openConnection(queryManagerURL, lifeCycleManagerURL);
		
		if (args[0].equalsIgnoreCase("LIST"))
		{
			if (args.length != 2)
			{
				System.out.println("Bitte Programm richtig aufrufen! (... LIST <serviceName>)");
			}
			else
			{
				registry.listWSDLs(args[1]);
			}
		}
		else if (args[0].equalsIgnoreCase("REGISTER"))
		{
			if (args.length != 4)
			{
				System.out.println("Bitte Programm richtig aufrufen! (... REGISTER <orgName> <serviceName> <wsdlURL>)");
			}
			else
			{
				registry.authenticate("gruppe1", "");
				registry.registerService(args[1], args[2], args[3]);
			}
		}
		else
		{
			System.out.println("Ungueltiger Programmaufruf.\nVersuche REGISTER oder LIST als parameter um weitere Informationen zu erhalten.");
		}
		
		System.out.println("Verbindung wird beendet...");
		registry.closeConnection();
	}
}
