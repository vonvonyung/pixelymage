import streamlit as st
# import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.util import img_as_float
from PIL import Image
import io



def convert_to_pixel_art(image, n_segments, compactness, sigma):
    image = img_as_float(image)
    segments = slic(image, n_segments=n_segments, compactness=compactness, sigma=sigma)
    segment_colors = {}
    for label in np.unique(segments):
        mask = np.where(segments == label)
        color = np.mean(image[mask], axis=0)
        segment_colors[label] = color

    result = np.zeros_like(image)
    for label in np.unique(segments):
        mask = np.where(segments == label)
        result[mask] = segment_colors[label]

    result = (result * 255).astype(np.uint8)
    return result

# 替代方案1：基于颜色量化的像素化（使用K-means）
def convert_to_pixel_art_kmeans(image, n_colors=16):
    from sklearn.cluster import KMeans
    h, w, channels = image.shape  # 动态获取通道数
    pixels = image.reshape(-1, channels)  # 适配任意通道数
    
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(pixels)
    new_pixels = kmeans.cluster_centers_[kmeans.labels_]
    
    return new_pixels.reshape(h, w, channels).astype(np.uint8)

# 方案：混合SLIC和K-means
def convert_to_pixel_hybrid(image, n_segments, n_colors):
    """混合算法：先SLIC分割再K-means颜色量化"""
    from sklearn.cluster import KMeans
    
    # SLIC预处理
    image_float = img_as_float(image)
    segments = slic(image_float, n_segments=n_segments, compactness=10)
    
    # 对每个超像素进行K-means
    result = np.zeros_like(image_float)
    for label in np.unique(segments):
        mask = segments == label
        pixels = image[mask]
        
        # 动态计算实际可用的颜色数量
        unique_colors = np.unique(pixels, axis=0)
        actual_n_colors = min(len(unique_colors), n_colors)
        
        if len(pixels) > actual_n_colors and actual_n_colors > 1:
            kmeans = KMeans(n_clusters=actual_n_colors, random_state=0).fit(pixels)
            pixels = kmeans.cluster_centers_[kmeans.labels_]
        
        result[mask] = pixels
    
    return (result * 255).astype(np.uint8)


# 替代方案2：固定区块马赛克效果
def convert_to_pixel_art_mosaic(image, block_size=8):
    h, w = image.shape[:2]
    # 更安全的尺寸调整方式
    new_h = (h // block_size) * block_size
    new_w = (w // block_size) * block_size
    cropped = image[:new_h, :new_w]
    
    # 适配任意通道数
    channels = image.shape[2] if len(image.shape) > 2 else 1
    if len(image.shape) == 2:  # 灰度图处理
        cropped = cropped[..., np.newaxis]
    
    # 创建马赛克
    mosaic = cropped.reshape(
        new_h//block_size, block_size,
        new_w//block_size, block_size,
        channels
    ).mean(axis=(1,3)).astype(np.uint8)
    
    # 处理单通道情况
    if channels == 1:
        mosaic = mosaic.squeeze()
    
    return np.repeat(np.repeat(mosaic, block_size, axis=0), block_size, axis=1)

# 替代方案3：基于分水岭算法的区域分割
def convert_to_pixel_art_watershed(image, marker_low=0.1, marker_high=0.9, sigma=1.0):
    from skimage.filters import sobel, gaussian
    from skimage.segmentation import watershed
    
    blurred = gaussian(image, sigma=sigma, channel_axis=-1)
    gradient = sobel(blurred.mean(axis=2))
    
    markers = np.zeros(gradient.shape, dtype=np.int32)
    markers[gradient < marker_low] = 1
    markers[gradient > marker_high] = 2
    
    # 修正参数名称
    labels = watershed(image=gradient, markers=markers, watershed_line=True)  # 修改此处参数名称
    
    result = np.zeros_like(image)
    for label in np.unique(labels):
        if label == 0:
            continue
        result[labels == label] = image[labels == label].mean(axis=0)
    
    return result.astype(np.uint8)


def color_replacement(image, target_color, new_color, tolerance=30):
    # 在函数开始添加调试信息
    print(f"目标颜色: {target_color}, 新颜色: {new_color}")
    print(f"图像尺寸: {image.shape}, 数据类型: {image.dtype}")
    
    
    
    # 增加透明度通道处理
    has_alpha = image.shape[-1] == 4 if image.ndim == 3 else False
    
    # 转换颜色格式时保留透明度
    target = np.array(target_color[:3])  # 保持RGB
    if has_alpha:
        new = np.append(new_color[:3], 255)  # 添加不透明alpha通道
    else:
        new = np.array(new_color[:3])
    
    # 根据通道数调整计算方式
    if image.ndim == 2:  # 灰度图
        diff = np.abs(image - target[0])
    elif has_alpha:  # 带透明度的彩色图
        diff = np.linalg.norm(image[..., :3] - target, axis=2)
    else:  # 普通彩色图
        diff = np.linalg.norm(image - target, axis=2)
    
    mask = diff <= tolerance
    result = image.copy()
    
    # 处理不同通道情况
    if has_alpha:
        result[mask] = new  # 同时替换RGBA
    else:
        result[mask] = new[:image.shape[-1]]  # 自动适配通道数
        
    replaced_pixels = np.count_nonzero(mask)
    print(f"替换了 {replaced_pixels} 个像素")
    return result
# 智能颜色替换函数
def smart_color_replace(pixel_art, target_color, new_color, blending=0.5):
    """基于颜色相似度的智能替换"""
    from sklearn.neighbors import NearestNeighbors
    
    # 获取颜色簇
    pixels = pixel_art.reshape(-1, 3)
    nbrs = NearestNeighbors(n_neighbors=5).fit(pixels)
    
    # 查找相似颜色
    distances, indices = nbrs.kneighbors([target_color])
    similar_colors = pixels[indices[0]]
    
    # 创建替换蒙版
    mask = np.isin(pixels, similar_colors).all(axis=1)
    mask = mask.reshape(pixel_art.shape[:2])
    
    # 混合颜色
    blend = (pixel_art * (1 - blending) + new_color * blending).astype(np.uint8)
    result = np.where(mask[..., None], blend, pixel_art)
    return result


#################################################
############### Streamlit应用 ###################
#################################################

st.set_page_config(page_title='图片转像素画', page_icon=':art:', layout='wide')
st.title('图片转像素画')

# 创建两列布局
left_col, right_col = st.columns([0.3, 0.7])

with left_col:
    uploaded_file = st.file_uploader(label="请选择一张图片", type=["jpg", "jpeg", "png"], help="仅支持jpg/jpeg/png格式，文件大小不超过200MB")
    if uploaded_file:
        # 当新文件上传时，重置相关状态
        if 'last_upload' not in st.session_state or st.session_state.last_upload != uploaded_file.name:
            st.session_state.pop('buffer', None)
            st.session_state.pop('pixel_art', None)
            st.session_state.last_upload = uploaded_file.name
        
        # 强制转换为RGB格式（修复RGBA问题）
        image = Image.open(uploaded_file)
        image = np.array(image)
        st.image(image, caption='原始图片', width=image.shape[1])

with right_col:
    if uploaded_file:
         # 算法选择器
        algorithm = st.selectbox(
            "选择像素化算法",
            ("SLIC超像素分割", "K-means颜色量化","马赛克效果", "分水岭区域分割"), 
            index=2
        )

        # 单层按钮布局
        btn_container = st.columns([0.8, 0.1, 0.1])  # [空白区域 | 保存按钮 | 下载按钮]

        # 动态参数区域 根据算法选择参数 
        if algorithm == "分水岭区域分割":  # 新增参数
            col1, col2 = st.columns(2)
            with col1:
                marker_low = st.slider('低标记阈值', 0.0, 2.0, 0.1, 0.01,
                                      help="值越低轮廓越多")
            with col2:
                marker_high = st.slider('高标记阈值', 0.0, 1.0, 0.9, 0.01,
                                       help="值越高细节保留越多")
            sigma = st.slider('高斯模糊强度', 0.0, 2.0, 1.0, 0.1,
                             help="消除图像噪声，值太大会丢失细节(低值保留细节，高值平滑图像)")
        elif algorithm == "SLIC超像素分割":
            n_segments = st.slider('超像素数量', 10, 5000, 100)
            compactness = st.slider('紧凑程度', 1, 50, 10)
            sigma = st.slider('高斯平滑标准差', 0.1, 5.0, 1.0)
        elif algorithm == "混合算法(SLIC+K-means)":  # 新增参数
            n_segments = st.slider('超像素数量', 10, 5000, 500)
            n_colors = st.slider('每区域颜色数', 2, 16, 3)
        elif algorithm == "马赛克效果":  # 新增参数控制
            block_size = st.slider('马赛克块大小', 2, 32, 8, 
                                 help="建议值：8-16（小图选小值，大图选大值）")
        else:  # K-means
            n_colors = st.slider('颜色数量', 2, 256, 16)

        # 生成像素画
        if algorithm == "混合算法(SLIC+K-means)":
            pixel_art = convert_to_pixel_hybrid(image, n_segments, n_colors)
        elif algorithm == "分水岭区域分割":
            pixel_art = convert_to_pixel_art_watershed(image, marker_low, marker_high, sigma)
        elif algorithm == "马赛克效果":
            pixel_art = convert_to_pixel_art_mosaic(image, block_size)
        elif algorithm == "SLIC超像素分割":
            pixel_art = convert_to_pixel_art(image, n_segments, compactness, sigma)
        else:
            pixel_art = convert_to_pixel_art_kmeans(image, n_colors)
            

        st.session_state['pixel_art'] = pixel_art
        st.image(pixel_art, caption='像素画', width=pixel_art.shape[1])

        # 颜色替换
        st.session_state['pixel_art'] = pixel_art
        display_img = st.image(pixel_art, caption='像素画', width=pixel_art.shape[1], use_container_width ='auto')


        # 新增颜色替换面板
        with st.expander("🎨 高级颜色替换"):
            col1, col2, col3 = st.columns([0.3, 0.3, 0.4])
            with col1:
                target_color = st.color_picker("选择要替换的颜色", "#FF0000")
            with col2:
                new_color = st.color_picker("替换为颜色", "#00FF00") 
            with col3:
                tolerance = st.slider("颜色容差", 0, 100, 30)
                replace_mode = st.selectbox("替换模式", ["精确替换", "智能混合"])
                if replace_mode == "智能混合":
                    blending = st.slider("颜色混合度", 0.0, 1.0, 0.5)
            
            if st.button("执行替换"):
                # 转换颜色格式时处理透明度
                target = np.array([int(target_color[1:3],16), 
                                 int(target_color[3:5],16),
                                 int(target_color[5:7],16)])
                                 
                # 根据图片是否包含alpha通道生成新颜色
                if st.session_state.pixel_art.shape[-1] == 4:
                    new = np.array([int(new_color[1:3],16),
                                  int(new_color[3:5],16),
                                  int(new_color[5:7],16),
                                  255])  # 添加完全不透明alpha
                else:
                    new = np.array([int(new_color[1:3],16),
                                  int(new_color[3:5],16),
                                  int(new_color[5:7],16)])
                
                # 执行替换
                replaced = color_replacement(st.session_state.pixel_art, 
                                           target, new, tolerance)
                st.session_state.pixel_art = replaced
                display_img.image(replaced)  # 直接更新显示对象
                st.session_state.buffer = None  # 清除旧缓存

                # st.session_state.refresh_flag = not st.session_state.get('refresh_flag', False)
                # display_img.image(replaced, caption=f'像素画 {st.session_state.refresh_flag}')



        # 修改保存和下载按钮逻辑
        with btn_container[1]:
            if st.button('保存图片') and 'pixel_art' in st.session_state:
                # 立即生成新的缓冲区数据
                img = Image.fromarray(st.session_state.pixel_art)
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                st.session_state.buffer = buffer.getvalue()  # 更新缓冲区

        # 下载按钮显示
        if 'pixel_art' in st.session_state and st.session_state.last_upload == uploaded_file.name:
            with btn_container[2]:
                # 实时生成下载数据
                img = Image.fromarray(st.session_state.pixel_art)
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                st.download_button(
                    label="点击下载",
                    data=buffer.getvalue(),
                    file_name="pixel_art.png",
                    mime="image/png"
                )