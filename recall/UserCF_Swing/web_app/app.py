"""
Flask Web应用 - UserCF Swing推荐系统可视化界面
"""
from flask import Flask, render_template, jsonify, request
import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

# 添加code目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))
from UserCF_Swing import UserCFSwingRecommender

app = Flask(__name__)

def convert_to_python_types(obj):
    """将numpy/pandas类型转换为Python原生类型，以便JSON序列化"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_types(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj

# 全局推荐器实例
recommender = None
algorithm_status = {
    'initialized': False,
    'data_loaded': False,
    'similarity_calculated': False,
    'current_step': '未开始'
}

def init_recommender():
    """初始化推荐器"""
    global recommender, algorithm_status
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        code_dir = os.path.join(script_dir, '..', 'code')
        data_dir = os.path.join(code_dir, 'data')
        
        recommender = UserCFSwingRecommender(data_dir, load_from_cache=True)
        
        # 确保加载了交互数据
        if recommender.interaction_df is None:
            interaction_path = os.path.join(data_dir, 'interaction_table.csv')
            if os.path.exists(interaction_path):
                recommender.interaction_df = pd.read_csv(interaction_path)
            else:
                print(f"警告: 交互数据文件不存在: {interaction_path}")
        
        # 确保user_df和item_df已加载
        if recommender.user_df is None:
            user_path = os.path.join(data_dir, 'user_table.csv')
            if os.path.exists(user_path):
                recommender.user_df = pd.read_csv(user_path)
        
        if recommender.item_df is None:
            item_path = os.path.join(data_dir, 'item_table.csv')
            if os.path.exists(item_path):
                recommender.item_df = pd.read_csv(item_path)
        
        algorithm_status['initialized'] = True
        algorithm_status['data_loaded'] = True
        algorithm_status['current_step'] = '数据加载完成'
        
        return True
    except Exception as e:
        import traceback
        error_msg = f"初始化失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return False

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/api/init', methods=['POST'])
def api_init():
    """初始化推荐系统"""
    global recommender, algorithm_status
    try:
        if init_recommender():
            return jsonify({
                'success': True,
                'message': '推荐系统初始化成功',
                'status': algorithm_status
            })
        else:
            return jsonify({
                'success': False,
                'message': '推荐系统初始化失败'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'初始化错误: {str(e)}'
        }), 500

@app.route('/api/status', methods=['GET'])
def api_status():
    """获取算法状态"""
    return jsonify(algorithm_status)

@app.route('/api/data/stats', methods=['GET'])
def api_data_stats():
    """获取数据统计信息"""
    global recommender
    if recommender is None:
        return jsonify({'error': '推荐系统未初始化'}), 400
    
    try:
        stats = {
            'num_users': len(recommender.user_df) if recommender.user_df is not None else 0,
            'num_items': len(recommender.item_df) if recommender.item_df is not None else 0,
            'num_interactions': len(recommender.interaction_df) if recommender.interaction_df is not None else 0,
            'avg_interactions_per_user': round(
                len(recommender.interaction_df) / len(recommender.user_df) if recommender.user_df is not None and len(recommender.user_df) > 0 else 0, 
                2
            )
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/users', methods=['GET'])
def api_users():
    """获取用户列表"""
    global recommender
    if recommender is None:
        return jsonify({'error': '推荐系统未初始化'}), 400
    
    try:
        if recommender.user_df is None:
            return jsonify({'users': []})
        
        users = recommender.user_df.to_dict('records')
        return jsonify({'users': users})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/calculate_similarity', methods=['POST'])
def api_calculate_similarity():
    """计算用户相似度"""
    global recommender, algorithm_status
    if recommender is None:
        return jsonify({'error': '推荐系统未初始化'}), 400
    
    try:
        use_weights = request.json.get('use_weights', True)
        return_steps = request.json.get('return_steps', True)
        
        # 如果相似度未计算，则计算
        if recommender.user_similarity is None:
            steps_info = recommender.calculate_user_similarity(use_weights=use_weights, return_steps=return_steps)
            algorithm_status['similarity_calculated'] = True
            algorithm_status['current_step'] = '相似度计算完成'
        else:
            steps_info = None
        
        # 统计相似度信息
        num_users_with_similarity = len(recommender.user_similarity) if recommender.user_similarity else 0
        
        result = {
            'success': True,
            'message': '相似度计算完成',
            'num_users': int(num_users_with_similarity),
            'status': algorithm_status
        }
        
        if steps_info:
            # 转换steps_info中的所有numpy类型
            result['calculation_steps'] = convert_to_python_types(steps_info)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/user/<int:user_id>/interactions', methods=['GET'])
def api_user_interactions(user_id):
    """获取用户交互历史"""
    global recommender
    if recommender is None:
        return jsonify({'error': '推荐系统未初始化'}), 400
    
    try:
        if recommender.interaction_df is None:
            return jsonify({'interactions': []})
        
        user_interactions = recommender.interaction_df[
            recommender.interaction_df['user_id'] == user_id
        ].head(10).to_dict('records')
        
        # 添加物品信息并转换numpy类型
        for interaction in user_interactions:
            item_id = int(interaction['item_id'])
            interaction['user_id'] = int(interaction['user_id'])
            interaction['item_id'] = item_id
            
            if recommender.item_df is not None:
                item_info = recommender.item_df[recommender.item_df['item_id'] == item_id]
                if not item_info.empty:
                    interaction['item_name'] = str(item_info['item_name'].iloc[0])
                    interaction['item_category'] = str(item_info['category'].iloc[0])
                    interaction['item_price'] = float(item_info['price'].iloc[0])
                else:
                    interaction['item_name'] = f'物品{item_id}'
                    interaction['item_category'] = '未知'
                    interaction['item_price'] = 0.0
            else:
                interaction['item_name'] = f'物品{item_id}'
                interaction['item_category'] = '未知'
                interaction['item_price'] = 0.0
            
            # 确保interaction_type是字符串
            if 'interaction_type' in interaction:
                interaction['interaction_type'] = str(interaction['interaction_type'])
        
        return jsonify({'interactions': user_interactions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/user/<int:user_id>/similar_users', methods=['GET'])
def api_similar_users(user_id):
    """获取相似用户"""
    global recommender
    if recommender is None:
        return jsonify({'error': '推荐系统未初始化'}), 400
    
    if recommender.user_similarity is None:
        return jsonify({'error': '相似度未计算'}), 400
    
    try:
        top_k = int(request.args.get('top_k', 5))
        
        # 检查用户是否在相似度矩阵中
        if user_id not in recommender.user_similarity:
            return jsonify({'similar_users': []})
        
        # 获取相似用户列表
        user_similarity_dict = recommender.user_similarity.get(user_id, {})
        
        # 如果相似用户字典为空，返回空列表
        if not user_similarity_dict:
            return jsonify({'similar_users': []})
        
        # 排序并取Top-K
        similar_users = sorted(
            user_similarity_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        result = []
        for sim_user_id, similarity in similar_users:
            try:
                # 确保sim_user_id是Python原生int类型
                sim_user_id = int(sim_user_id)
                similarity = float(similarity)
                
                # 检查用户数据是否存在
                if recommender.user_df is None:
                    result.append({
                        'user_id': sim_user_id,
                        'similarity': round(similarity, 4),
                        'age': '未知',
                        'gender': '未知'
                    })
                    continue
                
                user_info = recommender.user_df[recommender.user_df['user_id'] == sim_user_id]
                if not user_info.empty:
                    age_val = user_info['age'].iloc[0]
                    gender_val = user_info['gender'].iloc[0]
                    result.append({
                        'user_id': sim_user_id,
                        'similarity': round(similarity, 4),
                        'age': int(age_val) if pd.notna(age_val) else '未知',
                        'gender': str(gender_val) if pd.notna(gender_val) else '未知'
                    })
                else:
                    # 如果用户信息不存在，仍然返回相似度
                    result.append({
                        'user_id': sim_user_id,
                        'similarity': round(similarity, 4),
                        'age': '未知',
                        'gender': '未知'
                    })
            except Exception as e:
                # 如果单个用户处理失败，记录错误但继续处理其他用户
                print(f"处理用户 {sim_user_id} 时出错: {str(e)}")
                result.append({
                    'user_id': int(sim_user_id) if isinstance(sim_user_id, (np.integer, np.int64)) else sim_user_id,
                    'similarity': round(float(similarity), 4),
                    'age': '未知',
                    'gender': '未知'
                })
        
        return jsonify({'similar_users': result})
    except Exception as e:
        import traceback
        error_msg = f"获取相似用户失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({'error': str(e), 'details': traceback.format_exc()}), 500

@app.route('/api/user/<int:user_id>/recommendations', methods=['GET'])
def api_recommendations(user_id):
    """获取推荐结果"""
    global recommender
    if recommender is None:
        return jsonify({'error': '推荐系统未初始化'}), 400
    
    if recommender.user_similarity is None:
        return jsonify({'error': '相似度未计算'}), 400
    
    if recommender.interaction_df is None:
        return jsonify({'error': '交互数据未加载'}), 400
    
    if recommender.item_df is None:
        return jsonify({'error': '物品数据未加载'}), 400
    
    try:
        top_n = int(request.args.get('top_n', 10))
        return_steps = request.args.get('return_steps', 'true').lower() == 'true'
        
        # 检查用户是否有交互历史
        user_interactions = recommender.interaction_df[recommender.interaction_df['user_id'] == user_id]
        if user_interactions.empty:
            return jsonify({
                'error': f'用户 {user_id} 没有交互历史，无法生成推荐',
                'recommendations': []
            }), 400
        
        # 生成推荐（带计算步骤）
        if return_steps:
            recommendations, steps_info = recommender.recommend_items_with_reasons_and_steps(user_id, top_n)
        else:
            recommendations = recommender.recommend_items_with_reasons(user_id, top_n)
            steps_info = None
        
        # 如果没有推荐结果
        if not recommendations or len(recommendations) == 0:
            result = {
                'recommendations': [],
                'message': '未找到推荐结果，可能该用户没有相似用户或相似用户没有推荐物品'
            }
            if steps_info:
                result['calculation_steps'] = steps_info
            return jsonify(result)
        
        result = []
        for item_id, score, reason in recommendations:
            # 确保item_id是Python原生int类型
            item_id = int(item_id)
            
            item_info = recommender.item_df[recommender.item_df['item_id'] == item_id]
            if not item_info.empty:
                item_data = {
                    'item_id': int(item_id),
                    'item_name': str(item_info['item_name'].iloc[0]),
                    'category': str(item_info['category'].iloc[0]),
                    'price': float(item_info['price'].iloc[0]),
                    'score': round(float(score), 4),
                    'reason': {
                        'similar_users': [],
                        'common_items': []
                    }
                }
                
                # 添加相似用户信息
                for sim_user in reason.get('similar_users', [])[:3]:
                    sim_user_id = int(sim_user['user_id'])
                    sim_user_info = recommender.user_df[recommender.user_df['user_id'] == sim_user_id]
                    if not sim_user_info.empty:
                        item_data['reason']['similar_users'].append({
                            'user_id': int(sim_user_id),
                            'similarity': round(float(sim_user['similarity']), 4),
                            'contribution': round(float(sim_user['contribution']), 4),
                            'interaction_type': str(sim_user['interaction_type']),
                            'age': int(sim_user_info['age'].iloc[0]) if pd.notna(sim_user_info['age'].iloc[0]) else None,
                            'gender': str(sim_user_info['gender'].iloc[0]) if pd.notna(sim_user_info['gender'].iloc[0]) else None
                        })
                
                # 添加共同物品信息
                common_items = reason.get('common_items', [])
                for common_item_id in common_items[:3]:
                    common_item_id = int(common_item_id)
                    common_item_info = recommender.item_df[recommender.item_df['item_id'] == common_item_id]
                    if not common_item_info.empty:
                        item_data['reason']['common_items'].append({
                            'item_id': int(common_item_id),
                            'item_name': str(common_item_info['item_name'].iloc[0]),
                            'category': str(common_item_info['category'].iloc[0])
                        })
                
                result.append(item_data)
        
        response = {'recommendations': result}
        if steps_info:
            # 转换steps_info中的所有numpy类型
            response['calculation_steps'] = convert_to_python_types(steps_info)
        
        return jsonify(response)
    except Exception as e:
        import traceback
        error_msg = f"生成推荐失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({
            'error': str(e),
            'details': traceback.format_exc(),
            'recommendations': []
        }), 500

@app.route('/api/evaluate', methods=['POST'])
def api_evaluate():
    """评估推荐系统"""
    global recommender
    if recommender is None:
        return jsonify({'error': '推荐系统未初始化'}), 400
    
    if recommender.user_similarity is None:
        return jsonify({'error': '相似度未计算'}), 400
    
    try:
        test_ratio = float(request.json.get('test_ratio', 0.2))
        top_n = int(request.json.get('top_n', 10))
        
        results = recommender.evaluate_with_global_split(
            test_ratio=test_ratio,
            top_n=top_n,
            min_user_interactions=5
        )
        
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import pandas as pd
    # 初始化推荐器
    init_recommender()
    print("=" * 60)
    print("UserCF Swing 推荐系统 Web界面")
    print("=" * 60)
    print("访问 http://127.0.0.1:5000 查看可视化界面")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)

