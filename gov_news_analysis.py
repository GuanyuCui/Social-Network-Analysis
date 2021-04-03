import os
import re, string
# 分词
import jieba
import jieba.posseg as pseg
import thulac
import pkuseg
from LAC import LAC
# 网络分析
import networkx as nx
import networkx.algorithms.community as nx_comm
import community
# 表格相关
import pandas as pd
from tabulate import tabulate

def has_chinese(string):
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

# 从新闻文件中获得实体信息
# @param news_file: 需要提取的新闻文件
# @param save_file: 保存文件
# @param mode: 分词提取使用模式 in (jieba, pkuseg, thulac)
# @return entity_freq, set_news_people: 实体-词频字典 {entity: freq}, 新闻编号-人物集合字典 {count: set{people}}
def fetch_entities(news_file, save_file = 'default_entity_file.txt', mode = 'jieba'):
	if not os.path.exists(news_file):
		print('打不开新闻文件! 退出中...')
		return None
	
	# 文本
	texts = []
	with open(news_file) as f:
		# 去掉标题行
		headline = f.readline()
		# 读入每一行
		for line in f:
			# 五个数据域
			url, date, meta, title, text = line.split(sep = '\t')
			texts.append(text)
	f.close()
	del f, headline, line, url, date, meta, title, text
	
	# 保留标签
	# nr	人名
	# nrfg	人名(jieba)
	# nrt	外国人名(jieba)
	# np	人名(thulac)
	# PER	人名(baidu)
	# ns	地名
	# LOC	地名(baidu)
	# nt	机构团体
	# ni	机构名(thulac)
	# ORG	机构名(baidu)
	# nz	其他专名
	tags = ['nr', 'nrfg', 'nrt', 'np', 'PER', 'ns', 'LOC', 'nt', 'ni', 'ORG', 'nz']
	
	# 各人/地实体出现次数
	entity_freq = {}
	# 每条新闻出现的人
	set_news_people = {}
	
	# 第几条新闻
	text_count = 0
	# 共几条新闻
	text_total = len(texts)
	
	# 不同模式的执行语句不同
	if mode == 'jieba':
		cut_func_str = 'pseg.cut(text)'
	elif mode == 'pkuseg':
		seg = pkuseg.pkuseg(postag = True)
		seg
		cut_func_str = 'seg.cut(text)'
	elif mode == 'thulac':
		thu1 = thulac.thulac()
		thu1
		cut_func_str = 'thu1.cut(text)'
	elif mode == 'baidu':
		lac = LAC(mode = 'lac')
		lac
		cut_func_str = 'lac.run(text)'
	
	for text in texts:
		print('正处理新闻', text_count, '/', text_total, '...')
		# 去掉特殊转义字符
		text = re.sub(r'[\t\r\n]', ' ', text)
		# 去掉多余空格
		text = re.sub(r'[ ]+', ' ', text)
		text = text.strip()
		# 人实体
		people = set()
		# 分词加标签
		words_tags = eval(cut_func_str)
		# baidu 模式需要单独处理一下
		if mode == 'baidu':
			words_tags = [ (words_tags[0][i], words_tags[1][i]) for i in range( len(words_tags[0]) ) ]
		# 对每个词
		for word, tag in words_tags:
			# 去掉空格和除·以外的标点符号
			word = re.sub(r'[~!@#$%^&*()_+{}|:"<>?`=-\[\]\\;\',./～！¥…*（）—+「」：“”《》〈〉？【】、；‘’，。]', '', word)
			word = word.replace(' ', '')
			# 如果是人/地 (去掉单字)
			if tag in tags and len(word) > 1:
				# 增加词频
				if word in entity_freq:
					entity_freq[word] += 1
				else:
					entity_freq[word] = 1
				# 如果是人物
				if tag in tags[:5] and has_chinese(word):
					people.add(word)
		print(people)
		# 新闻序号 -> 人集合			
		set_news_people[text_count] = people
		text_count += 1
	print('保存处理结果...')
	#保存
	f = open(save_file, 'w')
		
	f.write(str(entity_freq) + '\n')
	f.write(str(set_news_people) + '\n')
	f.close()
	print('新闻实体处理完成!')
	return entity_freq, set_news_people

# 统计出现频率最高的实体
# @param entity_freq: 实体-词频字典
# @param top_k: 前 k 个
# @param return_tabulate: 是否返回表格
# @return 词频前 k 高的实体的表格 或 dict
def most_frequently_entities_topk(entity_freq, top_k = 10, return_tabulate = True):
	# 按值降序
	entity_freq = sorted(entity_freq.items(), key = lambda kv: kv[1], reverse = True)
	# 前 top_k 个
	topk_entities_list = entity_freq[: top_k]
	topk_entities_dict = {k : v for k, v in topk_entities_list}
	# 构建表格并返回
	if return_tabulate:
		df = pd.DataFrame(topk_entities_dict.items(), columns = ['实体', '词频'])
		return tabulate(df, headers = 'keys', tablefmt = 'psql')
	else:
		return topk_entities_dict
	
# 从之前获取的集合建立人物社交图
# @param set_news_people: 新闻序号-人物集合字典/列表
# @return G: 社交网络图
def build_social_network(set_news_people):
	print('社交网络构建中...')
	# 计算每条边的权重 {(p1, p2) : {'weight': xxx}}
	edges_dict = {}
	for i in range(len(set_news_people)):
		# 把人物实体集合转成列表
		people = list(set_news_people[i])
		# 选定两个人，增加他们之间的权重
		for i in range(len(people)):
			for j in range(i + 1, len(people)):
				if (people[i], people[j]) not in edges_dict.keys():
					edges_dict[(people[i], people
						[j])] = {'weight': 1}
				else:
					edges_dict[(people[i], people[j])]['weight'] += 1
	# 把字典形式的边转化成列表 [(p1, p2, {'weight': xxx})]
	edges = [(*k, v) for k, v in edges_dict.items()]
	# 建图
	G = nx.Graph()
	G.add_edges_from(edges)
	return G

# 做各种预处理工作
# @param news_file: 新闻文本
# @param entity_file: 实体文件
# @param network_file: 社交网络文件
# @return G: 社交网络图
def preprocess(news_file, entity_file = 'default_entity_file.txt', network_file = 'default_network_file.txt', cut_mode = 'jieba'):	
	print('(1) 分词与实体提取:')
	# 如果实体文件不存在
	if not os.path.exists(entity_file):
		entity_freq, set_news_people = fetch_entities(news_file, entity_file, mode = cut_mode)
	else:
		#读取预处理文件
		print('利用已经存在的文件', entity_file, '...')
		f = open(entity_file, 'r')
		l = f.readline()
		entity_freq = eval(l)
		l = f.readline()
		set_news_people = eval(l)
		f.close()
	print('分词与实体提取完成!')
	
	print('')
	print('(2) 热门人物/机构:')
	# 获取出现频率最高的若干实体
	k = 10
	print('出现频率最高的', k, '个实体为:')
	print(most_frequently_entities_topk(entity_freq, k))
	
	print('')
	print('(3) 社交网络构建:')
	# 如果网络文件不存在
	if not os.path.exists(network_file):
		G = build_social_network(set_news_people)
		nx.write_weighted_edgelist(G, network_file)
	else:
		# 读取预处理文件
		print('利用已经存在的文件', network_file, '...')
		G = nx.read_weighted_edgelist(network_file)
	print('社交网络构建完成!')
	return G

# 找寻某节点权重最高的邻居
# @param graph: 图
# @param node: 节点
# @param top_k: 权重前 k 大
# @param return_tabulate: 是否返回表格
# @return 权重前 k 大的邻居的表格 或 dict
def strongest_neighbors_topk(graph, node, top_k = 10, return_tabulate = True):
	# 没有该节点就不返回
	if node not in graph:
		print('图内没有该节点!')
		return None
	print('节点', node, '的邻居:')
	# 按 weight 降序排序邻接表的边
	# 并取前k个
	neighbor = sorted(graph[node].items(),
					key = lambda kv: kv[1]['weight'],
					reverse = True)
	topk_neighbor_list = neighbor[:top_k]
	# 变成 {'neighbor': weight} 的简单字典
	topk_neighbor_dict = {k : v['weight'] for k, v in topk_neighbor_list}
	# 构建表格并返回
	if return_tabulate:
		df = pd.DataFrame(topk_neighbor_dict.items(), columns = ['邻居', '权重'])
		return tabulate(df, headers = 'keys', tablefmt = 'psql')
	else:
		return topk_neighbor_dict

# 图的统计数据
# @param graph: 图
# @return nodes, edges, components, largest_comp: 图的节点数、边数、连通分量数以及最大连通分量节点数
def graph_statistics(graph):
	nodes = nx.number_of_nodes(graph)
	edges = nx.number_of_edges(graph)
	components = nx.number_connected_components(graph)
	largest_comp = len( max(nx.connected_components(graph), key = len) )
	print('节点个数:', nodes)
	print('边数:', edges)
	print('连通分量个数:', components)
	print('最大连通分量的大小:', largest_comp)
	return nodes, edges, components, largest_comp
	
# 影响力(pagerank) top_k
# @param graph: 图
# @param return_tabulate: 是否返回表格
# @return 节点影响力 top_k 表格 或字典
def pagerank_influence_topk(graph, top_k = 20, return_tabulate = True):
	# pagerank
	influence = sorted(nx.pagerank(graph).items(), key = lambda kv: kv[1], reverse = True)
	topk_influence_list = influence[:top_k]
	topk_influence_dict = {k : v for k, v in topk_influence_list}
	# 构建表格并返回
	if return_tabulate:
		df = pd.DataFrame(topk_influence_dict.items(), columns = ['节点', '影响力'])
		return tabulate(df, headers = 'keys', tablefmt = 'psql')
	else:
		return topk_influence_dict
	
# 基础分析
# @param graph: 社交网络图
def basic_analysis(graph):
	# 输入 A, 输出与 A 关系最强的 10 个邻居
	print('(1) 图的验证:')
	node = input('请输入节点名: ')
	# node = '习近平'
	print(strongest_neighbors_topk(graph, node = node))
	
	print('')
	# 图的节点个数、边数、连通分量个数、最大连通分量大小
	print('(2) 图的统计:')
	graph_statistics(graph)
	
	print('')
	# PageRank 最高的20个人
	print('(3) 影响力计算:')
	print(pagerank_influence_topk(graph, top_k = 20))

# (1) 小世界理论验证
# (未实现)
def smallworld_validate(graph):
	print('暂未实现')

# (2) 三元闭包验证
# (未实现)
def ternary_closure_validate(graph):
	print('暂未实现')

# (3) 社区挖掘
# @param graph: 社交网络图
# @param top_k: 打印前 k 个社区
# @param print_all: 是否打印社区内所有节点
# @param mode: 社区发现模式 (louvain/asyn_lpa/lpa)
# @return louvain 模式下返回 partition: {节点:所属社区}，其他两种模式返回社区生成器 partition_gen
def community_detection(graph, top_k = 5, print_all = False, mode = 'louvain'):
	# 先算所有节点的 pagerank
	pagerank_all = pagerank_influence_topk(graph, 
		graph.number_of_nodes(), return_tabulate = False)
	# louvain 算法
	if mode == 'louvain':
		partition = community.best_partition(graph)
		# 社区节点列表的列表
		comms = []
		for com in set(partition.values()):
			com_nodes_list = [node for node in partition.keys()
										if partition[node] == com]
			# 社区内节点按 pagerank 降序排列
			com_nodes_list = sorted(com_nodes_list, key = pagerank_all.get, reverse = True)
			comms.append(com_nodes_list)
		# 所有社区按节点数目降序排列
		comms = sorted(comms, key = len, reverse = True)
		print('在', mode, '社区发现模式下，共划分出', len(comms), '个社区')
		print('模块度为:', nx_comm.modularity(graph, comms))
		print()
		print('前', top_k, '个社区为:')
		print()
		# 打印前k个社区
		for i in range(top_k):
			if print_all:
				print('社区', i + 1, ':')
				print('大小:', len(comms[i]))
				print('节点:', comms[i])
			else:
				print('社区', i + 1, ':')
				print('大小:', len(comms[i]))
				print('代表元:', comms[i][0])
			print()
		return partition
	# Label propagation
	elif mode == 'asyn_lpa' or mode == 'lpa':
		if mode == 'asyn_lpa':
			partition_gen = nx_comm.asyn_lpa_communities(graph)
		else:
			partition_gen = nx_comm.label_propagation_communities(graph)
		comms = []
		for com in partition_gen:
			com_nodes_list = list(com)
			com_nodes_list = sorted(com_nodes_list, key = pagerank_all.get, reverse = True)
			comms.append(com_nodes_list)
		comms = sorted(comms, key = len, reverse = True)
		print('在', mode, '社区发现模式下，共划分出', len(comms), '个社区')
		print('模块度为:', nx_comm.modularity(graph, comms))
		print()
		print('前', top_k, '个社区为:')
		print()
		for i in range(top_k):
			if print_all:
				print('社区', i + 1, ':')
				print('大小:', len(comms[i]))
				print('节点:', comms[i])
			else:
				print('社区', i + 1, ':')
				print('大小:', len(comms[i]))
				print('代表元:', comms[i][0])
			print()
		return partition_gen
	else:
		print('社区发现模式不存在!')
		return None

# (4) 中介中心性计算
# @param graph: 社交网络图
# @param sample_ratio: 近似算法的采样比
# @param top_k: 前k
# @param return_tabulate: 是否返回表格
# @return 中心性 top_k 表格或字典
def centrality_topk(graph, sample_ratio = 0.05, top_k = 10, return_tabulate = True):
	centrality = sorted(
			nx.betweenness_centrality(graph, k = int(sample_ratio * graph.number_of_nodes()) ).items(),
			key = lambda kv: kv[1], reverse = True)
	topk_centrality_list = centrality[:top_k]
	topk_centrality_dict = {k : v for k, v in topk_centrality_list}
	# 构建表格并返回
	if return_tabulate:
		df = pd.DataFrame(topk_centrality_dict.items(), columns = ['节点', '中心性'])
		return tabulate(df, headers = 'keys', tablefmt = 'psql')
	else:
		return topk_centrality_dict

# (5) 聚集系数计算
# @param graph: 社交网络图
# @param top_k: 前k
# @param return_tabulate: 是否返回表格
# @return 聚集系数 top_k 表格 或字典
def clustering_coefficient_topk(graph, top_k = 10, return_tabulate = True):
	cc = sorted(nx.clustering(graph).items(), key = lambda kv: kv[1], reverse = True)
	topk_cc_list = cc[:top_k]
	topk_cc_dict = {k : v for k, v in topk_cc_list}
	# 构建表格并返回
	if return_tabulate:
		df = pd.DataFrame(topk_cc_dict.items(), columns = ['节点', '聚集系数'])
		return tabulate(df, headers = 'keys', tablefmt = 'psql')
	else:
		return topk_cc_dict

# (6) 结构洞挖掘
# (未实现)
def structural_holes_detection(graph):
	print('暂未实现')

# 可选分析
# @param graph: 社交网络图
def optional_analysis(graph):
	# 六选三
	print('(1) 小世界理论验证:')
	smallworld_validate(graph)
	
	print('')
	print('(2) 三元闭包验证:')
	ternary_closure_validate(graph)
	
	print('')
	print('(3) 社区挖掘:')
	community_detection(graph, top_k = 5, print_all = False, mode = 'louvain')
	community_detection(graph, top_k = 5, print_all = False, mode = 'asyn_lpa')
	community_detection(graph, top_k = 5, print_all = False, mode = 'lpa')

	print('')
	print('(4) （中介）中心性计算:')
	print(centrality_topk(graph, sample_ratio = 0.02, top_k = 10))
	
	print('')
	print('(5) 节点的聚集系数计算:')
	print(clustering_coefficient_topk(graph, top_k = 10))
	
	print('')
	print('(6) 结构洞挖掘:')
	structural_holes_detection(graph)

if __name__ == '__main__':
	print('一、数据预处理:')
	G = preprocess(news_file = './data/gov_news.txt', entity_file = './data/entities_data_baidu.txt', 
		network_file = './data/network_data_baidu.txt', cut_mode = 'baidu')
	nx.write_gexf(G, './data/graph.gexf')
	print('')
	print('二、基础分析:')
	basic_analysis(G)
	print('')
	print('三、可选分析:')
	optional_analysis(G)