import gradio as gr
from predictor import predict

def forecast(square_id, hour, day_of_week):
    result = predict(square_id, hour, day_of_week)
    if isinstance(result, str):
        return result
    return result

iface = gr.Interface(
    fn=forecast,
    inputs=[
        gr.Number(label="Square ID", value=123),
        gr.Slider(0, 23, step=1, label="Giờ trong ngày"),
        gr.Radio(["Thứ hai", "Thứ ba", "Thứ tư", "Thứ năm", "Thứ sáu", "Thứ bảy", "Chủ nhật"], label="Ngày trong tuần")
    ],
    outputs=gr.Dataframe(label="Dự báo 24 giờ (144 bước x 10 phút)"),
    title="Dự báo hoạt động mạng theo Square ID",
)

if __name__ == "__main__":
    iface.launch()