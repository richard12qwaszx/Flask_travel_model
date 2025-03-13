import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import networkx as nx
from flask import Flask, render_template, request, jsonify, send_file
import io
import base64
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib
from torch_geometric.data import Data
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import warnings
import matplotlib.font_manager as fm
import time
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import shortest_path
from collections import defaultdict
import math
import json

warnings.filterwarnings('ignore')
matplotlib.use('Agg')  # 使用非交互式後端

# 自定義JSON編碼器處理NumPy類型
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# 查找系統可用的中文字型
def find_chinese_font():
    # 常見的中文字型列表
    chinese_fonts = [
        'SimHei', 'Microsoft YaHei', '微軟正黑體', 'Microsoft JhengHei',
        'PingFang TC', 'PingFang SC', 'Noto Sans CJK TC', 'Noto Sans CJK SC',
        'Heiti TC', 'Heiti SC', 'WenQuanYi Micro Hei', 'Source Han Sans CN', 
        'Source Han Sans TW', 'Hiragino Sans GB'
    ]
    
    # 獲取所有系統字型
    font_paths = fm.findSystemFonts()
    system_fonts = [fm.get_font(font).family_name for font in font_paths]
    
    # 尋找可用的中文字型
    for font in chinese_fonts:
        if font in system_fonts:
            print(f"找到可用的中文字型: {font}")
            return [font]
    
    # 如果找不到中文字型，則返回空列表，使用matplotlib默認字型
    print("找不到中文字型，將使用預設字型。中文可能無法正確顯示。")
    return []

# 設置中文字型支持
chinese_font = find_chinese_font()
if chinese_font:
    plt.rcParams['font.sans-serif'] = chinese_font
    plt.rcParams['axes.unicode_minus'] = False
else:
    # 如果沒有找到中文字型，嘗試使用一個通用配置
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print("使用DejaVu Sans作為後備字型。中文可能無法正確顯示，但程式仍會繼續執行。")

# 定義與原始模型相同的GNN架構
class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

# 全局變數，用於存儲加載的模型和數據
global_model = None
global_df = None
global_nodes = None
global_node_embeddings = None
global_node_mapping = None
global_device = None
global_clusters = None
global_node_embeddings_2d = None
global_cache = {}
base_visualization = None
global_location_coords = None
global_distance_matrix = None

# 加載模型和數據
def load_model_and_data():
    global global_model, global_df, global_nodes, global_node_embeddings, global_node_mapping
    global global_device, global_clusters, global_node_embeddings_2d, global_cache
    global global_location_coords, global_distance_matrix
    
    print("讀取保存的模型和數據...")
    
    # 讀取原始數據 - 使用更高效的方式
    global_df = pd.read_csv('838-所有評論原始資料.csv', encoding='utf-8', usecols=['user_id', 'gmap_location', 'translated_comments', 'score'])
    
    # 預先計算每個位置的平均評分和評論數，避免重複計算
    location_stats = global_df.groupby('gmap_location').agg({
        'score': 'mean',
        'translated_comments': 'count'
    }).rename(columns={'translated_comments': 'review_count'})
    
    # 讀取保存的節點嵌入向量
    global_node_embeddings = np.load('node_embeddings.npy')
    
    # 重新創建地點之間的關係圖以獲取節點列表
    user_locations = global_df.groupby('user_id')['gmap_location'].apply(list)
    user_locations = user_locations[user_locations.apply(len) > 1]
    
    G = nx.DiGraph()
    for locations in user_locations:
        for i in range(len(locations) - 1):
            source = locations[i]
            target = locations[i+1]
            if G.has_edge(source, target):
                G[source][target]['weight'] += 1
            else:
                G.add_edge(source, target, weight=1)
    
    global_nodes = list(G.nodes())
    global_node_mapping = {node: i for i, node in enumerate(global_nodes)}
    
    # 預先計算每個位置的關鍵詞頻率
    location_keywords = {}
    for location in global_nodes:
        location_reviews = global_df[global_df['gmap_location'] == location]['translated_comments'].fillna('').values
        location_keywords[location] = ' '.join(location_reviews)
    
    # 初始化模型
    global_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global_model = GNN(in_channels=1, hidden_channels=16, out_channels=8).to(global_device)
    
    # 加載保存的模型權重
    try:
        global_model.load_state_dict(torch.load('gnn_model.pt', map_location=global_device))
        global_model.eval()  # 設置為評估模式
        print("模型加載成功！")
    except Exception as e:
        print(f"加載模型時出錯: {e}")
    
    # 使用KMeans計算聚類
    kmeans = KMeans(n_clusters=4, random_state=42)
    global_clusters = kmeans.fit_predict(global_node_embeddings)
    
    # 使用PCA降維以便可視化
    pca = PCA(n_components=2)
    global_node_embeddings_2d = pca.fit_transform(global_node_embeddings)
    
    # 將降維後的座標作為景點的虛擬地理座標
    global_location_coords = {}
    for i, node in enumerate(global_nodes):
        # 將NumPy類型轉換為Python原生類型
        global_location_coords[node] = (float(global_node_embeddings_2d[i, 0]), float(global_node_embeddings_2d[i, 1]))
    
    # 計算景點之間的距離矩陣
    location_points = np.array(list(global_location_coords.values()))
    dist_matrix = squareform(pdist(location_points, 'euclidean'))
    global_distance_matrix = pd.DataFrame(
        dist_matrix, 
        index=global_location_coords.keys(), 
        columns=global_location_coords.keys()
    )
    
    # 預先計算相似度矩陣
    similarity_matrix = np.zeros((len(global_nodes), len(global_nodes)))
    for i in range(len(global_nodes)):
        emb_i = global_node_embeddings[i]
        norm_i = np.linalg.norm(emb_i)
        for j in range(len(global_nodes)):
            if i != j:
                emb_j = global_node_embeddings[j]
                similarity = np.dot(emb_i, emb_j) / (norm_i * np.linalg.norm(emb_j) + 1e-8)
                similarity_matrix[i, j] = similarity
    
    # 計算每個節點的平均相似度（普及度指標）
    popularity_scores = np.mean(similarity_matrix, axis=1)
    
    # 將所有預計算的數據存儲在全局字典中
    global_cache = {
        'location_stats': location_stats,
        'location_keywords': location_keywords,
        'similarity_matrix': similarity_matrix,
        'popularity_scores': popularity_scores
    }
    
    print("數據和模型加載完成！")

# 使用加載的模型進行推薦
def recommend_by_text_loaded(user_input, top_n=5):
    """
    使用加載的模型和嵌入向量進行基於自然語言的景點推薦
    """
    # 定義關鍵詞映射
    keyword_mapping = {
        "玩水": ["水", "海", "海灘", "游泳", "浮潛", "衝浪", "溫泉", "瀑布", "河", "湖", "潭", "游泳池", "戲水"],
        "登山": ["山", "爬山", "登山", "健行", "步道", "森林", "自然", "風景", "登頂", "郊山", "高山"],
        "文化": ["廟宇", "寺廟", "古蹟", "歷史", "文化", "藝術", "博物館", "展覽", "廟", "文化財", "古城", "老街"],
        "美食": ["美食", "小吃", "夜市", "餐廳", "飲食", "特產", "小吃街", "美味", "餐點", "食物", "特色餐"],
        "購物": ["購物", "商場", "市場", "紀念品", "伴手禮", "夜市", "精品", "買", "商店街", "特產", "賣"],
        "放鬆": ["放鬆", "休閒", "溫泉", "SPA", "度假", "渡假村", "療癒", "舒緩", "舒適", "休息", "避暑"],
        "拍照": ["拍照", "風景", "美景", "景色", "打卡", "景點", "自拍", "攝影", "日落", "日出", "壯麗", "美麗"],
        "親子": ["親子", "兒童", "小孩", "家庭", "遊樂", "遊戲", "互動", "教育", "體驗", "適合小朋友", "適合家庭"],
    }
    
    # 提取輸入中的關鍵字
    input_keywords = []
    for category, keywords in keyword_mapping.items():
        for keyword in keywords:
            if keyword in user_input:
                if not any(k in input_keywords for k in keywords):
                    input_keywords.extend(keywords)
                break
    
    # 如果沒有找到特定關鍵字，提取所有非停用詞作為關鍵字
    if not input_keywords:
        # 簡單的中文停用詞
        stopwords = ["我", "想", "要", "去", "的", "是", "有", "在", "來", "能", "會", "可以", "哪裡", "推薦", "地方"]
        input_keywords = [word for word in user_input if word not in stopwords and len(word.strip()) > 0]
    
    # 去除重複關鍵詞
    input_keywords = list(set(input_keywords))
    
    # 計算每個景點與關鍵詞的匹配度 - 使用預先計算的緩存
    location_scores = {}
    for location in global_nodes:
        # 獲取預先計算的統計數據
        location_text = global_cache['location_keywords'].get(location, '')
        stats = global_cache['location_stats'].loc[location] if location in global_cache['location_stats'].index else {'score': 0, 'review_count': 0}
        
        # 關鍵詞匹配次數
        keyword_count = sum(1 for keyword in input_keywords if keyword in location_text)
        
        # 關鍵詞密度（考慮評論數量）
        review_count = stats['review_count']
        if review_count > 0:
            keyword_density = keyword_count / review_count
        else:
            keyword_density = 0
        
        # 獲取景點的嵌入向量
        loc_idx = global_node_mapping.get(location)
        if loc_idx is not None:
            # 獲取預先計算的普及度分數
            avg_similarity = global_cache['popularity_scores'][loc_idx]
            
            # 獲取評論評分
            avg_rating = stats['score']
            
            # 綜合評分：關鍵詞匹配度(0.5) + 景點普及度(0.2) + 評論評分(0.3)
            combined_score = 0.5 * keyword_density + 0.2 * avg_similarity + 0.3 * (avg_rating / 5.0)
            
            # 將結果存入字典 - 確保所有值都是Python原生類型
            location_scores[location] = {
                'score': float(combined_score),
                'keyword_matches': int(keyword_count),
                'review_count': int(review_count),
                'avg_rating': float(avg_rating),
                'popularity': float(avg_similarity)
            }
    
    # 排序並返回推薦結果
    sorted_locations = sorted(location_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    # 為最終結果添加詳細信息
    recommendations = []
    for location, data in sorted_locations[:top_n]:
        recommendations.append({
            'location': location,
            'score': float(data['score']),
            'keyword_matches': int(data['keyword_matches']),
            'review_count': int(data['review_count']), 
            'avg_rating': float(data['avg_rating']),
            'reason': generate_recommendation_reason(location, input_keywords, data),
            'coordinates': [float(c) for c in global_location_coords[location]]
        })
    
    return recommendations, input_keywords

# 生成推薦原因的輔助函數
def generate_recommendation_reason(location, keywords, data):
    """生成景點被推薦的原因說明"""
    reason = f"{location}很適合您，因為"
    
    if data['keyword_matches'] > 0:
        keyword_phrase = "、".join(keywords[:3]) if len(keywords) > 3 else "、".join(keywords)
        reason += f"在{data['review_count']}則評論中有{data['keyword_matches']}次提到與「{keyword_phrase}」相關的內容，"
    
    # 添加評分信息
    if data['avg_rating'] >= 4.5:
        reason += f"評價極高（{data['avg_rating']:.1f}/5分），"
    elif data['avg_rating'] >= 4.0:
        reason += f"評價優良（{data['avg_rating']:.1f}/5分），"
    elif data['avg_rating'] >= 3.5:
        reason += f"評價良好（{data['avg_rating']:.1f}/5分），"
    
    # 添加普及度信息
    if data['popularity'] > 0.6:
        reason += "且非常受到其他遊客歡迎。"
    elif data['popularity'] > 0.4:
        reason += "且相當受到遊客喜愛。"
    else:
        reason += "是個不錯的選擇。"
        
    return reason

# 規劃最佳路線
def plan_optimal_route(locations):
    """使用貪婪算法規劃景點之間的最佳路線"""
    if not locations:
        return []
    
    # 提取位置名稱列表
    location_names = [loc['location'] for loc in locations]
    
    # 如果只有一個景點，直接返回
    if len(location_names) <= 1:
        return location_names
    
    # 使用貪婪算法找到最短路徑
    current = location_names[0]  # 從第一個景點開始
    unvisited = set(location_names[1:])
    route = [current]
    
    # 依次找到最近的下一個景點
    while unvisited:
        next_location = min(unvisited, key=lambda x: global_distance_matrix.loc[current, x])
        route.append(next_location)
        unvisited.remove(next_location)
        current = next_location
    
    return route

# 預先生成基本圖形，只在需要時添加推薦點
def initialize_visualization():
    global base_visualization
    
    plt.figure(figsize=(12, 8))
    
    # 繪製所有景點（淡化顯示）
    for i in range(len(global_nodes)):
        plt.scatter(
            global_node_embeddings_2d[i, 0], 
            global_node_embeddings_2d[i, 1],
            s=100, 
            c='lightgray', 
            alpha=0.3
        )
    
    plt.title(f'推薦景點視覺化', fontsize=18, pad=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # 將基本圖形保存為 BytesIO 對象
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    
    base_visualization = img.getvalue()

# 修改後的可視化函數，添加路線規劃
def generate_visualization(recommendations):
    # 如果推薦數量為0，直接返回基本圖形
    if not recommendations:
        return base64.b64encode(base_visualization).decode()
    
    # 創建新的圖形
    plt.figure(figsize=(12, 8))
    
    # 繪製所有景點（淡化顯示）
    for i in range(len(global_nodes)):
        plt.scatter(
            global_node_embeddings_2d[i, 0], 
            global_node_embeddings_2d[i, 1],
            s=100, 
            c='lightgray', 
            alpha=0.3
        )
    
    colors = ['#4DBEEE', '#A2142F', '#77AC30', '#7E2F8E']
    markers = ['o', 's', '^', 'd']
    
    # 規劃最佳路線
    route = plan_optimal_route(recommendations)
    
    # 繪製路線
    if len(route) > 1:
        route_x = []
        route_y = []
        for loc_name in route:
            loc_idx = global_node_mapping[loc_name]
            route_x.append(global_node_embeddings_2d[loc_idx, 0])
            route_y.append(global_node_embeddings_2d[loc_idx, 1])
        
        plt.plot(route_x, route_y, 'r-', linewidth=2, alpha=0.7, zorder=1)
        
        # 添加箭頭指示方向
        for i in range(len(route_x) - 1):
            dx = route_x[i+1] - route_x[i]
            dy = route_y[i+1] - route_y[i]
            plt.arrow(
                route_x[i] + 0.7 * dx, 
                route_y[i] + 0.7 * dy,
                0.1 * dx, 0.1 * dy,
                head_width=0.15, 
                head_length=0.2, 
                fc='blue', 
                ec='blue',
                zorder=2
            )
    
    # 繪製推薦景點
    for i, loc_name in enumerate(route):
        idx = global_node_mapping[loc_name]
        cluster_id = global_clusters[idx]
        
        # 找到對應的推薦信息
        rec_info = next((r for r in recommendations if r['location'] == loc_name), None)
        
        plt.scatter(
            global_node_embeddings_2d[idx, 0], 
            global_node_embeddings_2d[idx, 1],
            s=300, 
            c=colors[cluster_id % len(colors)], 
            alpha=1.0,
            marker=markers[cluster_id % len(markers)],
            edgecolor='black',
            linewidth=2,
            zorder=3
        )
        
        # 添加序號標籤
        plt.annotate(
            f"{i+1}. {loc_name}",
            (global_node_embeddings_2d[idx, 0], global_node_embeddings_2d[idx, 1]),
            fontsize=14,
            ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.9),
            zorder=4
        )
    
    plt.title(f'推薦景點路線規劃', fontsize=18, pad=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # 將圖形轉換為base64編碼的圖像
    img_result = io.BytesIO()
    plt.savefig(img_result, format='png')
    img_result.seek(0)
    plt.close()
    
    return base64.b64encode(img_result.getvalue()).decode()

# 創建Flask應用
app = Flask(__name__)
app.json_encoder = NumpyEncoder  # 使用自定義的JSON編碼器

# 添加簡單的緩存機制
query_cache = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    start_time = time.time()
    user_query = request.form.get('query', '')
    if not user_query.strip():
        return jsonify({'error': '請輸入有效的查詢內容'})
    
    # 檢查緩存
    cache_key = user_query.strip().lower()
    if cache_key in query_cache:
        print(f"緩存命中: {cache_key}")
        response = query_cache[cache_key]
        response['from_cache'] = True
        response['processing_time'] = f"{time.time() - start_time:.3f}秒"
        return jsonify(response)
    
    try:
        recommendations, keywords = recommend_by_text_loaded(user_query, top_n=5)
        
        # 規劃路線
        route = plan_optimal_route(recommendations)
        
        # 生成可視化
        img_data = generate_visualization(recommendations)
        
        # 計算總路程和預估時間
        total_distance = 0
        if len(route) > 1:
            for i in range(len(route) - 1):
                total_distance += float(global_distance_matrix.loc[route[i], route[i+1]])
        
        # 假設每單位距離需要30分鐘
        estimated_time = total_distance * 30  # 分鐘
        
        # 生成行程安排
        itinerary = []
        current_time = 9 * 60  # 從早上9點開始，以分鐘為單位
        
        for i, loc_name in enumerate(route):
            # 找到對應的推薦信息
            rec_info = next((r for r in recommendations if r['location'] == loc_name), None)
            
            # 計算停留時間（根據景點評分和普及度）
            if rec_info:
                popularity = rec_info.get('popularity', 0.5)
                rating = rec_info.get('avg_rating', 3.0)
                stay_time = int(30 + (popularity * 30) + (rating * 10))  # 至少30分鐘，最多約2小時
            else:
                stay_time = 60  # 默認1小時
            
            # 計算到達時間和離開時間
            arrival_time = current_time
            departure_time = arrival_time + stay_time
            
            # 格式化時間
            arrival_time_str = f"{int(arrival_time // 60):02d}:{int(arrival_time % 60):02d}"
            departure_time_str = f"{int(departure_time // 60):02d}:{int(departure_time % 60):02d}"
            
            itinerary.append({
                'order': i + 1,
                'location': loc_name,
                'arrival_time': arrival_time_str,
                'departure_time': departure_time_str,
                'stay_time': f"{int(stay_time // 60)}小時{int(stay_time % 60)}分鐘"
            })
            
            # 更新當前時間，加上交通時間
            current_time = departure_time
            if i < len(route) - 1:
                travel_distance = float(global_distance_matrix.loc[route[i], route[i+1]])
                travel_time = int(travel_distance * 30)  # 每單位距離30分鐘
                current_time += travel_time
        
        response = {
            'recommendations': recommendations,
            'keywords': keywords,
            'visualization': img_data,
            'route': route,
            'total_distance': float(total_distance),
            'estimated_time': f"{int(estimated_time // 60)}小時{int(estimated_time % 60)}分鐘",
            'itinerary': itinerary,
            'from_cache': False,
            'processing_time': f"{time.time() - start_time:.3f}秒"
        }
        
        # 存入緩存
        query_cache[cache_key] = response
        
        return jsonify(response)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"處理請求時發生錯誤: {error_details}")
        return jsonify({'error': f'處理請求時發生錯誤: {str(e)}'})

# 添加健康檢查端點
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'ok', 
        'model_loaded': global_model is not None,
        'data_loaded': global_df is not None,
        'cache_size': len(query_cache),
        'memory_usage': {
            'df_size': f"{global_df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB" if global_df is not None else "N/A",
            'embeddings_size': f"{global_node_embeddings.nbytes / (1024 * 1024):.2f} MB" if global_node_embeddings is not None else "N/A"
        }
    })

# 添加緩存清理端點
@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    query_cache.clear()
    return jsonify({'status': 'ok', 'message': '緩存已清除'})

# 主函數
if __name__ == "__main__":
    try:
        # 加載模型和數據
        load_model_and_data()
        
        # 初始化基本可視化
        initialize_visualization()
        
        print(f"使用設備: {global_device}")
        print(f"資料集大小: {len(global_df)} 筆評論")
        print(f"景點數量: {len(global_nodes)} 個")
        
        # 啟動Flask應用
        print("啟動 Flask 應用程式...")
        app.run(debug=False, host='0.0.0.0', port=5001, threaded=True)
    except Exception as e:
        print(f"啟動應用程式時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
