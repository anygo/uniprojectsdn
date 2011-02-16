package mw.facebook;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

import javax.annotation.PostConstruct;
import javax.jws.WebMethod;
import javax.jws.WebService;
import javax.jws.soap.SOAPBinding;

import mw.facebook.graph.MWCCFriendCount;
import mw.facebook.graph.MWCCNeighborhood;
import mw.facebook.graph.MWCCNeighborhoodUtils;
import mw.facebook.graph.MWCCRealName;
import mw.facebook.graph.MWCCUniqueNode;
import mw.facebook.graph.MWCCUniqueNodeManager;

@WebService(name = "MWMyFacebookService", serviceName = "MWFacebookService")
@SOAPBinding(style = SOAPBinding.Style.RPC)
public class MWMyFacebookService implements MWFacebookServiceInterface {
	
	
	private LinkedList<Long> queries = new LinkedList<Long>();
	
	// ################
	// # INIT METHODS #
	// ################

	@PostConstruct
	void init() {
		System.out.println("INIT: " + getClass().getSimpleName());
		importRealNames("/proj/i4mw/data/names.list");
		importFriends("/proj/i4mw/data/friends.list");
		importFriendCounts("/proj/i4mw/data/friendcounts.list");
		System.out.println("COMPLETE: all data loaded");
	}

	private void importRealNames(String dataPath) {
		try {
			File file = new File(dataPath);
			BufferedReader reader = new BufferedReader(new FileReader(file));
			
			String line;
			while((line = reader.readLine()) != null) {
				String[] data = line.split("\t");
				MWCCUniqueNode newUniqueNode = MWCCUniqueNodeManager.getUniqueNode(data[0]);

				MWCCRealName realName = new MWCCRealName(data[1]);
				if("1509453326".equals(data[0])) {
					realName = new MWCCRealName("Gratulation! Dies scheint der richtige Weg zu sein.");
				}
				
				realName.addData(newUniqueNode);
			}
			System.out.println("COMPLETE: all real-name information loaded (currently " + MWCCUniqueNodeManager.listUniqueNodes().size() + " nodes total)");
			
			reader.close();
		} catch(Exception e) {
			e.printStackTrace();
			System.exit(1);
		}
	}

	private void importFriends(String dataPath) {
		try {
			// Create neighborhoods
			for(MWCCUniqueNode node: MWCCUniqueNodeManager.listUniqueNodes()) {
				MWCCNeighborhood newNeighborhood = new MWCCNeighborhood(node);
				newNeighborhood.addData(node);
			}

			File file = new File(dataPath);
			BufferedReader reader = new BufferedReader(new FileReader(file));
			String currentID = null;
			MWCCUniqueNode node = null;
			String line;

			// Create friend links
			while((line = reader.readLine()) != null) {
				String[] data = line.split("\t");
				if(!data[0].equals(currentID)) {
					node = MWCCUniqueNodeManager.getUniqueNode(data[0]);
					currentID = data[0];
				}
				MWCCUniqueNode neighbor = MWCCUniqueNodeManager.getUniqueNode(data[1]);
				MWCCNeighborhoodUtils.createNeighborLink(node, neighbor);
			}
			System.out.println("COMPLETE: all friends information loaded (currently " + MWCCUniqueNodeManager.listUniqueNodes().size() + " nodes total)");
			
			reader.close();
		} catch(Exception e) {
			e.printStackTrace();
			System.exit(1);
		}
	}
	
	// #######################
	// # WEB-SERVICE METHODS #
	// #######################
	
	@WebMethod
	public String getName(String id) throws MWUnknownIDException {
		
		queries.add(System.currentTimeMillis());
		if(!MWCCUniqueNodeManager.existsUniqueNode(id)) throw new MWUnknownIDException("UNKNOKN: id " + id);
		MWCCUniqueNode node = MWCCUniqueNodeManager.getUniqueNode(id);	
		return MWCCRealName.getData(node).getRealName();
	}

	@WebMethod
	public String[] getFriends(String id) throws MWUnknownIDException {
		
		queries.add(System.currentTimeMillis());
		if(!MWCCUniqueNodeManager.existsUniqueNode(id)) throw new MWUnknownIDException("UNKNOKN: id " + id);

		MWCCUniqueNode node = MWCCUniqueNodeManager.getUniqueNode(id);
		Collection<MWCCUniqueNode> friends = MWCCNeighborhoodUtils.getNeighbors(node, 1);
		
		String[] friendIDs = new String[friends.size()];
		int i = 0;
		for(MWCCUniqueNode friend: friends) {
			friendIDs[i++] = friend.getID();
		}

		return friendIDs;
	}

	@WebMethod
	public String[][] getFriendsBatch(String[] ids) throws MWUnknownIDException {
		
		queries.add(System.currentTimeMillis());
		String[][] friendIDs = new String[ids.length][];
		
		for(int i = 0; i < friendIDs.length; i++) {
			friendIDs[i] = getFriends(ids[i]);
		}

		return friendIDs;
	}

	@WebMethod
	public String[] searchIDs(String name) {
		
		queries.add(System.currentTimeMillis());
		List<String> result = new LinkedList<String>();
		
		Pattern pattern;
		try {
			pattern = Pattern.compile(name);
		} catch(PatternSyntaxException pse) {
			System.err.println("BAD PATTERN: \"" + name + "\"");
			return new String[0];
		}
		
		System.out.println("SEARCH: \"" + name + "\"");
		
		for(MWCCUniqueNode node: MWCCUniqueNodeManager.listUniqueNodes()) {
			String realName = MWCCRealName.getData(node).getRealName();
			Matcher matcher = pattern.matcher(realName);
			if(matcher.find()) {
				result.add(node.getID());
			}
		}
		
		return result.toArray(new String[result.size()]);
	}
	

	// ############################
	// # EXTENDED SERVICE METHODS #
	// ############################

	// Make this method 'public' to provide it via web service
	@WebMethod
	private String[] calculatePath(String idA, String idB) {
		
		queries.add(System.currentTimeMillis());
		if(!MWCCUniqueNodeManager.existsUniqueNode(idA)) return null;
		if(!MWCCUniqueNodeManager.existsUniqueNode(idB)) return null;
		
		MWCCUniqueNode nodeA = MWCCUniqueNodeManager.getUniqueNode(idA);
		MWCCUniqueNode nodeB = MWCCUniqueNodeManager.getUniqueNode(idB);
		List<MWCCUniqueNode> path = MWCCNeighborhoodUtils.getPath(nodeA, nodeB);
		
		String[] pathIDs = new String[path.size()];
		int i = 0;
		for(MWCCUniqueNode node: path) {
			pathIDs[i++] = node.getID();
		}
		return pathIDs;
	}

	// Make this method 'public' to provide it via web service
	@WebMethod
	private String[] getCircleOfFriends(String id, int depth) {
		
		queries.add(System.currentTimeMillis());
		MWCCUniqueNode node = MWCCUniqueNodeManager.getUniqueNode(id);
		Collection<MWCCUniqueNode> neighbors = MWCCNeighborhoodUtils.getNeighbors(node, depth);
		
		String[] neighborIDs = new String[neighbors.size()];
		int i = 0;
		for(MWCCUniqueNode neighbor: neighbors) {
			neighborIDs[i++] = neighbor.getID();
		}
		return neighborIDs;
	}

	
	// ################
	// # FRIEND COUNT #
	// ################

	private void importFriendCounts(String dataPath) {
		try {
			File file = new File(dataPath);
			BufferedReader reader = new BufferedReader(new FileReader(file));
			
			String line;
			while((line = reader.readLine()) != null) {
				String[] data = line.split("\t");
				MWCCUniqueNode newUniqueNode = MWCCUniqueNodeManager.getUniqueNode(data[0]);
				MWCCFriendCount friendCount = new MWCCFriendCount(Integer.parseInt(data[1]));
				friendCount.addData(newUniqueNode);
			}
			System.out.println("COMPLETE: all friend-count information loaded (currently " + MWCCUniqueNodeManager.listUniqueNodes().size() + " nodes total)");
			
			reader.close();
		} catch(Exception e) {
			e.printStackTrace();
			System.exit(1);
		}
	}

	// Make this method 'public' to provide it via web service
	@WebMethod
	private String getMostPopular(int n) {
		
		queries.add(System.currentTimeMillis());
		List<MWCCUniqueNode> nodesList = new ArrayList<MWCCUniqueNode>(MWCCUniqueNodeManager.listUniqueNodes().size());
		nodesList.addAll(MWCCUniqueNodeManager.listUniqueNodes());
		MWCCFriendCountComparator comparator = new MWCCFriendCountComparator();
		Collections.sort(nodesList, comparator);
		
		String result = "";
		for(int i = 0; i < n; i++) {
			MWCCUniqueNode node = nodesList.get(i);
			result += String.format("%4d %-40s %-40s\n", MWCCFriendCount.getData(node).getFriendCount(), node.getID(), MWCCRealName.getData(node).getRealName());
		}
		return result;
	}

	private static final class MWCCFriendCountComparator implements Comparator<MWCCUniqueNode> {

		public int compare(MWCCUniqueNode nodeA, MWCCUniqueNode nodeB) {
			MWCCFriendCount nodeAFriendCount = MWCCFriendCount.getData(nodeA);
			MWCCFriendCount nodeBFriendCount = MWCCFriendCount.getData(nodeB);
			
			if((nodeAFriendCount != null) && (nodeBFriendCount != null)) {
				return (nodeBFriendCount.getFriendCount() - nodeAFriendCount.getFriendCount());
			} else if((nodeAFriendCount == null) && (nodeBFriendCount == null)) {
				return 0;
			} else if(nodeAFriendCount == null) {
				return Integer.MAX_VALUE;
			} else {
				return Integer.MIN_VALUE;
			}
		}
		
	}

	
	
	@WebMethod
	public int getServerStatus(int interval) {

		long from = System.currentTimeMillis() - interval*1000;
		int count = 0;
		for(Iterator<Long> it = queries.descendingIterator(); it.hasNext(); ) {
			if(it.next() > from)
				count++;
			else
				break;
		}
		return count;
	}

}
