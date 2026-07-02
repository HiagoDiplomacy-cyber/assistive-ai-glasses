import os
# Corrige bug do MSMF no Windows que trava a captura de frames da câmera
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import threading
import time
import queue
import numpy as np
from PIL import Image

# Tenta carregar variáveis de ambiente do .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Controle de importação do Gemini SDK
gemini_disponivel = False
try:
    from google import genai
    gemini_disponivel = True
except ImportError:
    print("[Aviso]: Biblioteca 'google-genai' não encontrada. Rodando em modo simulação de IA.")

# Controle de importação do SpeechRecognition
speech_rec_disponivel = False
try:
    import speech_recognition as sr
    speech_rec_disponivel = True
except ImportError:
    print("[Aviso]: Biblioteca 'SpeechRecognition' não encontrada. Comandos de voz desativados.")

# Controle de importação do PyTTSx3 (Voz local)
pyttsx3_disponivel = False
try:
    import pyttsx3
    pyttsx3_disponivel = True
except ImportError:
    print("[Aviso]: Biblioteca 'pyttsx3' não encontrada. Áudio local desativado.")

# Controle de importação do YOLOv8
yolo_disponivel = False
try:
    from ultralytics import YOLO
    yolo_disponivel = True
except ImportError:
    print("[Aviso]: Biblioteca 'ultralytics' não encontrada. Detecção local YOLOv8 desativada.")

# --- CONFIGURAÇÕES E ESTADOS GLOBAIS ---
modo_rua = False
loja_procurada = None
contexto_usuario = None  # ex: "fome", "sede"
force_analysis = False
current_frame = None
last_gemini_desc = "Aguardando primeira análise..."
speech_history = []
yolo_boxes = []

# Fila de sintetização de voz (Threads seguras)
speech_queue = queue.Queue()

# --- SINTETIZADOR DE VOZ (THREAD WORKER) ---
def speech_worker():
    if not pyttsx3_disponivel:
        return
    try:
        engine = pyttsx3.init()
        # Configura voz em português
        voices = engine.getProperty('voices')
        for voice in voices:
            if "brazil" in voice.name.lower() or "portuguese" in voice.lang.lower() or "pt-br" in voice.id.lower():
                engine.setProperty('voice', voice.id)
                break
        engine.setProperty('rate', 190)  # Velocidade moderadamente rápida para acessibilidade
        
        while True:
            text = speech_queue.get()
            if text is None:
                break
            engine.say(text)
            engine.runAndWait()
            speech_queue.task_done()
    except Exception as e:
        print(f"[Erro de Voz]: Não foi possível iniciar o sintetizador pyttsx3: {e}")

if pyttsx3_disponivel:
    threading.Thread(target=speech_worker, daemon=True).start()

def speak(text):
    global speech_history
    if not text:
        return
    print(f"[Yuki de Ouvido]: {text}")
    
    # Atualiza histórico de áudio para o HUD (máximo de 3 linhas)
    speech_history.append(text)
    if len(speech_history) > 3:
        speech_history.pop(0)

    if pyttsx3_disponivel:
        # Limpa anúncios anteriores acumulados na fila para não acumular atrasos
        while not speech_queue.empty():
            try:
                speech_queue.get_nowait()
                speech_queue.task_done()
            except queue.Empty:
                break
        speech_queue.put(text)

# --- RECONHECIMENTO DE COMANDOS DE VOZ (THREAD) ---
def listen_commands_thread():
    global modo_rua, loja_procurada, contexto_usuario, force_analysis
    if not speech_rec_disponivel:
        return

    # Tenta obter microfone
    try:
        mic = sr.Microphone()
    except Exception as e:
        print(f"[Erro de Mic]: Sem acesso ao microfone. Comandos por voz desativados: {e}")
        return

    r = sr.Recognizer()
    with mic as source:
        r.adjust_for_ambient_noise(source, duration=1.0)

    print("[Voz]: Ouvindo comandos por voz em segundo plano...")
    
    while True:
        try:
            with mic as source:
                audio = r.listen(source, phrase_time_limit=4)
            
            # Reconhecimento via API do Google em português
            text = r.recognize_google(audio, language="pt-BR").lower()
            print(f"[Escutado]: {text}")

            if "yuki" in text:
                if "ativar modo rua" in text or "modo rua" in text:
                    modo_rua = True
                    speak("Modo rua ativado. Focando em calçadas, obstáculos e placas de lojas.")
                elif "desativar modo rua" in text or "modo normal" in text or "desativar rua" in text:
                    modo_rua = False
                    speak("Modo normal ativado.")
                elif "estou com fome" in text or "fome" in text:
                    contexto_usuario = "fome"
                    speak("Entendido. Procurando locais de alimentação.")
                elif "estou com sede" in text or "sede" in text:
                    contexto_usuario = "sede"
                    speak("Entendido. Procurando água ou bebidas.")
                elif "limpar memória" in text or "esquecer" in text:
                    contexto_usuario = None
                    loja_procurada = None
                    speak("Memória limpa.")
                elif "procurando" in text or "procurando a" in text or "procurando o" in text:
                    parts = text.split("procurando")
                    if len(parts) > 1:
                        target = parts[1].strip()
                        # Limpa conectivos comuns
                        for word in ["a ", "o ", "por ", "uma ", "um "]:
                            if target.startswith(word):
                                target = target[len(word):].strip()
                        loja_procurada = target
                        speak(f"Buscando a loja {loja_procurada}.")
                elif "o que" in text and "frente" in text:
                    speak("Analisando o que está à frente.")
                    force_analysis = True
        except sr.UnknownValueError:
            pass  # Áudio não inteligível
        except sr.RequestError as e:
            print(f"[Erro de Conexão de Voz]: {e}")
            time.sleep(2)
        except Exception as e:
            print(f"[Erro no loop de voz]: {e}")

if speech_rec_disponivel:
    threading.Thread(target=listen_commands_thread, daemon=True).start()

# --- ANÁLISE COGNITIVA MULTIMODAL GEMINI (THREAD) ---
def gemini_vision_thread():
    global force_analysis, modo_rua, loja_procurada, contexto_usuario, last_gemini_desc
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_disponivel or not api_key or api_key == "SUA_API_KEY_AQUI":
        print("[Aviso]: GEMINI_API_KEY não configurada ou biblioteca ausente. Yuki rodando em modo Simulação Local.")
        last_gemini_desc = "Modo de Simulação Ativo (Insira a API Key para análise real)."
        
        while True:
            if force_analysis:
                force_analysis = False
                desc = "Simulação: Vejo uma calçada livre à frente e uma placa de cafeteria a 3 metros."
                if contexto_usuario == "fome":
                    desc += " Lembrei que você estava com fome. A cafeteria à frente pode ser uma opção."
                if loja_procurada:
                    desc += f" Ainda não encontrei a loja '{loja_procurada}'."
                speak(desc)
                last_gemini_desc = desc
            time.sleep(1)
        return

    # Inicializa o cliente oficial do Gemini SDK
    try:
        client = genai.Client(api_key=api_key)
        print("[Gemini]: API do Gemini conectada com sucesso!")
    except Exception as e:
        print(f"[Erro de Conexão Gemini]: {e}")
        last_gemini_desc = "Falha ao conectar com o Gemini API."
        return

    while True:
        # Espera de 8 segundos ou até que o comando de voz "forçar análise" seja chamado
        for _ in range(80):
            if force_analysis:
                break
            time.sleep(0.1)

        force_analysis = False
        
        global current_frame
        if current_frame is None:
            continue

        # Copia o frame atual para evitar problemas de concorrência com a captura OpenCV
        frame_copy = current_frame.copy()
        
        # Converte BGR para RGB e depois para imagem PIL
        rgb_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        modo = "RUA" if modo_rua else "NORMAL"
        loja_alvo = loja_procurada if loja_procurada else "Nenhuma no momento"
        necessidade = contexto_usuario if contexto_usuario else "Nenhuma no momento"

        prompt = f"""
        Você é a Yuki, a inteligência artificial de óculos assistivos para pessoas cegas.
        Seu papel é ser os olhos do usuário.
        
        INFORMAÇÕES DE CONTEXTO ATUAIS:
        - Modo de Operação: {modo} (No modo RUA, foque 100% em segurança, calçadas, carros, pessoas vindo na direção, semáforos, obstáculos terrestres e leitura de placas de lojas/restaurantes).
        - Loja Procurada pelo usuário: {loja_alvo} (Se a placa desta loja estiver visível, avise imediatamente).
        - Necessidade declarada do usuário: {necessidade} (Se for 'fome' ou 'sede', e você vir locais de alimentação/bebida, alerte proativamente).

        REGRAS DE RESPOSTA (EM PORTUGUÊS):
        1. Descreva o ambiente à frente de forma direta, clara e curta.
        2. Estime distâncias para obstáculos.
        3. Se vir uma escada, conte quantos degraus ela possui (ex: "escada com 6 degraus").
        4. Responda de forma fluida e conversacional. Não use formatação markdown, apenas texto puro.
        """

        try:
            print("[Yuki Gemini]: Analisando cena atual...")
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=[pil_image, prompt]
            )
            
            desc = response.text.strip()
            # Remove asteriscos ou marcações markdown que a voz possa ler estranho
            desc = desc.replace("*", "").replace("#", "")
            
            last_gemini_desc = desc
            speak(desc)
        except Exception as e:
            print(f"[Erro Gemini API]: {e}")
            last_gemini_desc = "Erro ao processar imagem na API do Gemini."

threading.Thread(target=gemini_vision_thread, daemon=True).start()

# --- LOOP PRINCIPAL (OPENCV HUD & YOLO) ---
def main():
    global current_frame, yolo_boxes, modo_rua, loja_procurada, contexto_usuario, force_analysis
    
    yolo_model = None
    if yolo_disponivel:
        try:
            yolo_model = YOLO("yolov8n.pt")
            print("[YOLO]: Modelo YOLOv8 inicializado para detecção local.")
        except Exception as e:
            print(f"[Erro YOLO]: Falha ao carregar yolov8n.pt: {e}")

    # No Windows, o backend padrão MSMF costuma dar erro de grabFrame (-1072875772).
    # Tentamos primeiro DirectShow (cv2.CAP_DSHOW) que é muito mais estável.
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[Aviso Câmera]: Não foi possível abrir com CAP_DSHOW. Tentando padrão...")
        cap = cv2.VideoCapture(0)
        
    if not cap.isOpened():
        print("[Erro Câmera]: Não foi possível acessar a webcam local.")
        return

    try:
        print("\n=======================================================")
        print("Yuki Assistive Glasses iniciada com sucesso!")
        print("Comandos de voz suportados (diga 'Yuki' primeiro):")
        print("  - 'Yuki, ativar modo rua'")
        print("  - 'Yuki, desativar modo rua' ou 'Yuki, modo normal'")
        print("  - 'Yuki, estou com fome' ou 'Yuki, estou com sede'")
        print("  - 'Yuki, estou procurando a loja [nome]'")
        print("  - 'Yuki, o que está na minha frente?'")
        print("=======================================================\n")
        
        speak("Yuki inicializada e conectada. Câmera ativa.")

        prev_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Câmera]: Falha ao ler frame.")
                break
                
            current_frame = frame
            
            annotated_frame = frame.copy()
            if yolo_model:
                results = yolo_model(frame, verbose=False)
                if results and len(results) > 0:
                    annotated_frame = results[0].plot()

            h, w, _ = annotated_frame.shape
            
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (280, h), (15, 10, 25), -1)
            cv2.addWeighted(overlay, 0.75, annotated_frame, 0.25, 0, annotated_frame)
            
            cv2.line(annotated_frame, (280, 0), (280, h), (60, 60, 80), 2)
            
            cv2.putText(annotated_frame, "YUKI ASSIST VISOR", (15, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
            cv2.putText(annotated_frame, "Status: ATIVA", (15, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1)

            mic_status = "ONLINE" if speech_rec_disponivel else "OFFLINE"
            mic_color = (0, 255, 100) if speech_rec_disponivel else (100, 100, 255)
            cv2.putText(annotated_frame, f"Microfone: {mic_status}", (15, 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, mic_color, 1)
            
            modo_str = "RUA (Seguranca)" if modo_rua else "NORMAL (Essencia)"
            modo_color = (0, 165, 255) if modo_rua else (255, 180, 0)
            cv2.putText(annotated_frame, "MODO ATUAL:", (15, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(annotated_frame, modo_str, (15, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, modo_color, 2)
            
            necessidade_str = f"FOME ({contexto_usuario.upper()})" if contexto_usuario else "Nenhuma registrada"
            nec_color = (255, 80, 180) if contexto_usuario else (170, 170, 170)
            cv2.putText(annotated_frame, "CONTEXTO ATUAL:", (15, 165), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(annotated_frame, necessidade_str, (15, 185), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, nec_color, 1)

            loja_str = f"{loja_procurada}" if loja_procurada else "Nenhum alvo"
            loja_color = (0, 215, 255) if loja_procurada else (170, 170, 170)
            cv2.putText(annotated_frame, "BUSCANDO LOJA:", (15, 220), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(annotated_frame, loja_str, (15, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, loja_color, 1)

            cv2.putText(annotated_frame, "SALA DE VOZ (FONE):", (15, 285), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            y_offset = 305
            for history_text in speech_history:
                words = history_text.split()
                line = ""
                for word in words:
                    if len(line + " " + word) > 30:
                        cv2.putText(annotated_frame, line, (15, y_offset), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                        y_offset += 15
                        line = word
                    else:
                        line = line + " " + word if line else word
                if line:
                    cv2.putText(annotated_frame, line, (15, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                    y_offset += 20

            cv2.rectangle(annotated_frame, (290, h - 70), (w - 10, h - 10), (20, 20, 20), -1)
            cv2.rectangle(annotated_frame, (290, h - 70), (w - 10, h - 10), (0, 220, 255), 1)
            
            desc_words = last_gemini_desc.split()
            desc_line1 = ""
            desc_line2 = ""
            for word in desc_words:
                if len(desc_line1 + " " + word) < 55:
                    desc_line1 = desc_line1 + " " + word if desc_line1 else word
                else:
                    desc_line2 = desc_line2 + " " + word if desc_line2 else word
                    
            cv2.putText(annotated_frame, f"Yuki: {desc_line1}", (305, h - 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            if desc_line2:
                if len(desc_line2) > 55:
                    desc_line2 = desc_line2[:52] + "..."
                cv2.putText(annotated_frame, desc_line2, (305, h - 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if curr_time - prev_time > 0 else 30
            prev_time = curr_time
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (w - 90, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1)

            cv2.imshow("Yuki Assistive Glasses Dashboard", annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == ord('r'):
                modo_rua = not modo_rua
                status_str = "ativado" if modo_rua else "desativado"
                speak(f"Modo rua {status_str} manualmente.")
            elif key == ord('f'):
                contexto_usuario = "fome" if contexto_usuario != "fome" else None
                status_str = "Fome ativada" if contexto_usuario else "Memória limpa"
                speak(status_str)
            elif key == ord('s'):
                speak("Forçando análise visual imediata.")
                force_analysis = True
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Yuki finalizada.")
        os._exit(0)

if __name__ == "__main__":
    main()
