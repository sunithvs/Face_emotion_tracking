import cv2
from fer import FER


def main():
    detector = FER(mtcnn=True)

    emotion_peoples = {}
    img = cv2.imread("data/img.png")
    count = 0
    for emotions in detector.detect_emotions(img):
        count += 1
        e_max, e_name = 0, ''
        emotions = emotions['emotions']

        for emotion in emotions:
            if emotions[emotion] > e_max:
                e_max, e_name = emotions[emotion], emotion

        emotion_peoples[e_name] = emotion_peoples[e_name] + 1 if e_name in emotion_peoples else 1

    e_max, e_name = 0, ''

    print("\n\n-------------------------------------------------------------------\n\n")
    print(f"There are {count}  people out of them")

    for emotion in emotion_peoples:
        eno = emotion_peoples[emotion]
        e_max, e_name = (eno, emotion) if eno > e_max else (e_max, e_name)

        print(f"=> {eno} {'people' if eno > 1 else 'person'} feeling {emotion}")

    print(f"\nOn average the photo looks {e_name}")


if __name__ == "__main__":
    main()
