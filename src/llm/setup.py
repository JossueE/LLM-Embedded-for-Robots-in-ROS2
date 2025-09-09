from setuptools import find_packages, setup

package_name = 'llm'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/llm.launch.py']),
        ('share/' + package_name + '/config', ['config/models.yml']),
        ('share/' + package_name + '/scripts', ['scripts/download_models.sh']), 
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='snorlix',
    maintainer_email='kadavetar10@icloud.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'llm_main = llm.llm_main:main',
            'audio_listener = llm.audio_listener:main',
            'audio_publisher = llm.audio_publisher:main',
            'wake_word = llm.wake_word_detector:main',
            'stt = llm.speech_to_text:main',
            'tts = llm.text_to_speech:main'
        ],
    },
)
