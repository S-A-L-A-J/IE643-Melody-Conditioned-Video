"""Local-only IE643 test interface. Run with the source project's Python environment."""
import gradio as gr
from inference import generate


def run(audio, start, remove_vocals, model_variant, phase, duration, progress=gr.Progress()):
    try:
        video, report, plot, metadata = generate(audio,start,remove_vocals,
                                                progress=lambda label:progress(0,desc=label),
                                                model_variant=model_variant,visual_phase=phase,duration=duration)
    except (ValueError, RuntimeError) as error:
        raise gr.Error(str(error)) from error
    timing = report['timing']
    score = f"{timing['f1']:.3f}" if timing else 'Unavailable: no detected beats'
    summary = (f"**Beat-event F1:** {score} · **Runtime:** {report['runtime_seconds']:.1f} s\n\n"
               'Each visual event can match one estimated audio beat within 125 ms. This score does not measure visual quality or mood.')
    return video,summary,plot,metadata


with gr.Blocks(title='Nullspace · Music to Video') as app:
    gr.Markdown('# Music to Video\nUpload music to generate a four- or eight-second abstract video. '
                'The trained model follows musical changes to generate moving rings and colours.')
    audio = gr.Audio(type='filepath',label='Song or instrumental',sources=['upload'])
    start = gr.Number(value=0,minimum=0,label='Start time (seconds)')
    remove = gr.Checkbox(value=True,label='Suppress singing before generation (disable for an instrumental)')
    model_variant=gr.Radio(choices=[('Improved temporal model','temporal'),('Earlier baseline','baseline')],
                           value='temporal',label='Model comparison')
    phase=gr.Slider(0,360,value=0,step=5,label='Visual arrangement (degrees)')
    duration=gr.Radio(choices=[4,8],value=4,label='Clip duration (seconds)')
    button = gr.Button('Generate video',variant='primary')
    video = gr.Video(label='Generated video',format='mp4',width=360,height=360)
    summary = gr.Markdown()
    plot = gr.Image(label='Music and visual timing',type='filepath')
    metadata = gr.File(label='Download result details')
    gr.Markdown('Output: 4 or 8 seconds · 128 × 128 · 8 fps. Eight seconds uses the improved temporal model. '
                'The improved model has passed validation timing checks. Independent testing and mood evaluation remain pending.')
    button.click(run,[audio,start,remove,model_variant,phase,duration],[video,summary,plot,metadata],concurrency_limit=1)

if __name__ == '__main__':
    app.queue(max_size=4).launch(server_name='127.0.0.1',server_port=7863,share=False,
                               allowed_paths=[str(__import__('pathlib').Path(__file__).resolve().parents[1]/'demo_outputs')])
