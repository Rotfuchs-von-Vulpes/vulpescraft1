#include <stdio.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <cglm/cglm.h>

unsigned int VBO;
unsigned int cubeVAO, lightVAO;

const char *cubeFShader =
		"#version 330 core\n"
		"out vec4 FragColor;\n"
		"void main()\n"
		"{\n"
		" FragColor=vec4(1.);\n"
		"}";
const char *cubeVShader =
		"#version 330 core\n"
		"layout(location=0)in vec3 aPos;\n"
		"uniform mat4 model;\n"
		"uniform mat4 view;\n"
		"uniform mat4 projection;\n"
		"void main()\n"
		"{\n"
		" gl_Position=projection*view*model*vec4(aPos,1.);\n"
		"}";
const char *lightCasterFShader =
		"#version 330 core\n"
		"out vec4 FragColor;\n"
		"struct Material{\n"
		" float shininess;\n"
		"};\n"
		"struct Light{\n"
		" vec3 direction;\n"
		" vec3 ambient;\n"
		" vec3 diffuse;\n"
		" vec3 specular;\n"
		"};\n"
		"in vec3 FragPos;\n"
		"in vec3 Normal;\n"
		"in vec2 TexCoords;\n"
		"uniform vec3 viewPos;\n"
		"uniform Material material;\n"
		"uniform sampler2D materialDiffuse;\n"
		"uniform sampler2D materialSpecular;\n"
		"uniform Light light;\n"
		"void main()\n"
		"{\n"
		" vec3 ambient=light.ambient*texture(materialDiffuse,TexCoords).rgb;\n"
		" vec3 norm=normalize(Normal);\n"
		" vec3 lightDir=normalize(-light.direction);\n"
		" float diff=max(dot(norm,lightDir),0.);\n"
		" vec3 diffuse=light.diffuse*diff*texture(materialDiffuse,TexCoords).rgb;\n"
		" vec3 viewDir=normalize(viewPos-FragPos);\n"
		" vec3 reflectDir=reflect(-lightDir,norm);\n"
		" float spec=pow(max(dot(viewDir,reflectDir),0.),material.shininess);\n"
		" vec3 specular=light.specular*spec*texture(materialSpecular,TexCoords).rgb;\n"
		" vec3 result=ambient+diffuse+specular;\n"
		" FragColor=vec4(result,1.);\n"
		"}";
const char *lightCasterVShader =
		"#version 330 core\n"
		"layout(location=0)in vec3 aPos;\n"
		"layout(location=1)in vec3 aNormal;\n"
		"layout(location=2)in vec2 aTexCoords;\n"
		"out vec3 FragPos;\n"
		"out vec3 Normal;\n"
		"out vec2 TexCoords;\n"
		"uniform mat4 model;\n"
		"uniform mat4 view;\n"
		"uniform mat4 projection;\n"
		"void main()\n"
		"{\n"
		" FragPos=vec3(model*vec4(aPos,1.));\n"
		" Normal=mat3(transpose(inverse(model)))*aNormal;\n"
		" TexCoords=aTexCoords;\n"
		" gl_Position=projection*view*vec4(FragPos,1.);\n"
		"}";

const char *vertexShaderSource =
		"#version 330 core\n"
		"layout (location = 0) in vec3 aPos;\n"
		"layout (location = 1) in vec2 aTexCoord;\n"
		"out vec2 TexCoord;\n"
		"uniform mat4 model;\n"
		"uniform mat4 view;\n"
		"uniform mat4 projection;\n"
		"void main()\n"
		"{\n"
		"  gl_Position = projection * view * model * vec4(aPos, 1.0);\n"
		"  TexCoord = aTexCoord.xy;\n"
		"}\0";

const char *fragmentShaderSource =
		"#version 330 core\n"
		"out vec4 FragColor;\n"
		"in vec2 TexCoord;\n"
		"uniform sampler2D texture1;\n"
		"void main()\n"
		"{\n"
		" FragColor = texture(texture1, TexCoord);\n"
		"}";

unsigned int shaderProgram[2];
unsigned int texture1, texture2, texture3;
unsigned int texture;
int textureCount = 1;
int shaderCount = 0;

double lastTime;
int nbFrames = 0;
int fps = 0;
char title[100];

int screenWidth = 600;
int screenHeight = 600;

void loadShader(unsigned int vertexShader, unsigned int fragmentShader)
{
	shaderProgram[shaderCount] = glCreateProgram();

	glAttachShader(shaderProgram[shaderCount], vertexShader);
	glAttachShader(shaderProgram[shaderCount], fragmentShader);
	glLinkProgram(shaderProgram[shaderCount]);

	glUseProgram(shaderProgram[shaderCount]);

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
	char infoLog[512];
	glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
	printf("%s\n", infoLog);

	++shaderCount;
}

unsigned int loadTexture(const char *image)
{
	glGenTextures(++textureCount, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	// set the texture wrapping parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT); // set texture wrapping to GL_REPEAT (default wrapping method)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	// set texture filtering parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	// load image, create texture and generate mipmaps
	int width, height, nrChannels;
	stbi_set_flip_vertically_on_load(true);
	unsigned char *data = stbi_load(image, &width, &height, &nrChannels, 0);
	if (data)
	{
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
		glGenerateMipmap(GL_TEXTURE_2D);
	}
	else
	{
		puts("Failed to load texture");
	}
	stbi_image_free(data);

	return texture;
}

mat4 view = GLM_MAT4_IDENTITY_INIT;
mat4 projection;

vec3 cameraPos = {0.0f, 0.0f, 3.0f};
vec3 cameraFront = {0.0f, 0.0f, -1.0f};
vec3 cameraUp = {0.0f, 1.0f, 0.0f};

float deltaTime = 0.0f; // time between current frame and last frame
float lastFrame = 0.0f;

void resizeCallback(GLFWwindow *window, int width, int height)
{
	screenWidth = width;
	screenHeight = height;
	glViewport(0, 0, screenWidth, screenHeight);
	glm_perspective(45.0f, (float)screenWidth / (float)screenHeight, 0.1f, 100.0f, projection);
}

void processInput(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	float cameraSpeed = (2.5 * deltaTime);
	vec3 scale;
	vec3 cross;
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
	{
		glm_vec3_scale(cameraFront, cameraSpeed, scale);
		glm_vec3_add(cameraPos, scale, cameraPos);
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
	{
		glm_vec3_scale(cameraFront, -cameraSpeed, scale);
		glm_vec3_add(cameraPos, scale, cameraPos);
	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
	{
		glm_cross(cameraFront, cameraUp, cross);
		glm_normalize(cross);
		glm_vec3_scale(cross, -cameraSpeed, scale);
		glm_vec3_add(cameraPos, scale, cameraPos);
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
	{
		glm_cross(cameraFront, cameraUp, cross);
		glm_normalize(cross);
		glm_vec3_scale(cross, cameraSpeed, scale);
		glm_vec3_add(cameraPos, scale, cameraPos);
	}
}

bool firstMouse = true;
float yaw = -90.0f;
float pitch = 0.0f;
float lastX = 800.0f / 2.0;
float lastY = 600.0 / 2.0;
float fov = 45.0f;

void mouseCallback(GLFWwindow *window, double xposIn, double yposIn)
{
	float xpos = xposIn;
	float ypos = yposIn;

	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top
	lastX = xpos;
	lastY = ypos;

	float sensitivity = 0.1f; // change this value to your liking
	xoffset *= sensitivity;
	yoffset *= sensitivity;

	yaw += xoffset;
	pitch += yoffset;

	// make sure that when pitch is out of bounds, screen doesn't get flipped
	if (pitch > 89.0f)
		pitch = 89.0f;
	if (pitch < -89.0f)
		pitch = -89.0f;

	vec3 front;
	front[0] = cos(glm_rad(yaw)) * cos(glm_rad(pitch));
	front[1] = sin(glm_rad(pitch));
	front[2] = sin(glm_rad(yaw)) * cos(glm_rad(pitch));
	glm_normalize(front);
	glm_vec3_copy(front, cameraFront);
}

void scrollCallback(GLFWwindow *window, double xoffset, double yoffset)
{
	fov -= (float)yoffset;
	if (fov < 1.0f)
		fov = 1.0f;
	if (fov > 45.0f)
		fov = 45.0f;
}

vec3 cubePositions[] = {
		{0.0f, 0.0f, 0.0f},
		{2.0f, 5.0f, -15.0f},
		{-1.5f, -2.2f, -2.5f},
		{-3.8f, -2.0f, -12.3f},
		{2.4f, -0.4f, -3.5f},
		{-1.7f, 3.0f, -7.5f},
		{1.3f, -2.0f, -2.5f},
		{1.5f, 2.0f, -2.5f},
		{1.5f, 0.2f, -1.5f},
		{-1.3f, 1.0f, -1.5f}};

double currentTime;

void drawCube(int index)
{
		mat4 model = GLM_MAT4_IDENTITY_INIT;
		glm_translate(model, cubePositions[index]);
		float angle = 20.0f * index;
		glm_rotate(model, 50.0f / 360 * 3.14 * (float)currentTime + angle, (vec3){0.5f, 1.0f, 0.0f});
		glUseProgram(shaderProgram[0]);
		unsigned int transformLoc = glGetUniformLocation(shaderProgram[0], "model");
		glUniformMatrix4fv(transformLoc, 1, false, (float *)model);

		glDrawArrays(GL_TRIANGLES, 0, 36);
}

void init()
{
	float vertices[] = {
			// positions          // normals           // texture coords
			-0.5f, -0.5f, -0.5f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f,
			0.5f, -0.5f, -0.5f, 0.0f, 0.0f, -1.0f, 1.0f, 0.0f,
			0.5f, 0.5f, -0.5f, 0.0f, 0.0f, -1.0f, 1.0f, 1.0f,
			0.5f, 0.5f, -0.5f, 0.0f, 0.0f, -1.0f, 1.0f, 1.0f,
			-0.5f, 0.5f, -0.5f, 0.0f, 0.0f, -1.0f, 0.0f, 1.0f,
			-0.5f, -0.5f, -0.5f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f,

			-0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
			0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f,
			0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
			0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
			-0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f,
			-0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,

			-0.5f, 0.5f, 0.5f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
			-0.5f, 0.5f, -0.5f, -1.0f, 0.0f, 0.0f, 1.0f, 1.0f,
			-0.5f, -0.5f, -0.5f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
			-0.5f, -0.5f, -0.5f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
			-0.5f, -0.5f, 0.5f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
			-0.5f, 0.5f, 0.5f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f,

			0.5f, 0.5f, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
			0.5f, 0.5f, -0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f,
			0.5f, -0.5f, -0.5f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
			0.5f, -0.5f, -0.5f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
			0.5f, -0.5f, 0.5f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
			0.5f, 0.5f, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,

			-0.5f, -0.5f, -0.5f, 0.0f, -1.0f, 0.0f, 0.0f, 1.0f,
			0.5f, -0.5f, -0.5f, 0.0f, -1.0f, 0.0f, 1.0f, 1.0f,
			0.5f, -0.5f, 0.5f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f,
			0.5f, -0.5f, 0.5f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f,
			-0.5f, -0.5f, 0.5f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f,
			-0.5f, -0.5f, -0.5f, 0.0f, -1.0f, 0.0f, 0.0f, 1.0f,

			-0.5f, 0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,
			0.5f, 0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f,
			0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
			0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
			-0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
			-0.5f, 0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};

	glEnable(GL_DEPTH_TEST);

	glGenVertexArrays(1, &cubeVAO);
	glGenBuffers(1, &VBO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindVertexArray(cubeVAO);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(6 * sizeof(float)));
	glEnableVertexAttribArray(2);

	glGenVertexArrays(1, &lightVAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);

	glBindVertexArray(lightVAO);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)0);
	glEnableVertexAttribArray(0);

	unsigned int vertexShader;
	vertexShader = glCreateShader(GL_VERTEX_SHADER);

	glShaderSource(vertexShader, 1, &lightCasterVShader, NULL);
	glCompileShader(vertexShader);

	unsigned int fragmentShader;
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	glShaderSource(fragmentShader, 1, &lightCasterFShader, NULL);
	glCompileShader(fragmentShader);

	loadShader(vertexShader, fragmentShader);

	vertexShader = glCreateShader(GL_VERTEX_SHADER);

	glShaderSource(vertexShader, 1, &cubeVShader, NULL);
	glCompileShader(vertexShader);

	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	glShaderSource(fragmentShader, 1, &cubeFShader, NULL);
	glCompileShader(fragmentShader);

	loadShader(vertexShader, fragmentShader);

	texture1 = loadTexture("../src/default_box.png");
	texture2 = loadTexture("../src/default_iron_frame.png");
	texture3 = loadTexture("../src/default_sand.png");

	glUseProgram(shaderProgram[0]);
	glUniform1i(glGetUniformLocation(shaderProgram[0], "materialDiffuse"), 0);
	glUniform1i(glGetUniformLocation(shaderProgram[0], "materialSpecular"), 1);

	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);

	glm_translate(view, (vec3){0.0f, 0.0f, -3.0f});
}

void render()
{
	currentTime = glfwGetTime();

	vec3 center;
	glm_vec3_add(cameraPos, cameraFront, center);
	glm_lookat(cameraPos, center, cameraUp, view);
	glm_perspective(glm_rad(fov), screenWidth / screenHeight, 0.1f, 100.0f, projection);

	glUseProgram(shaderProgram[0]);

	unsigned int uniformLoc = glGetUniformLocation(shaderProgram[0], "light.direction");
	glUniform3f(uniformLoc, -0.2f, -1.0f, -0.3f);
	uniformLoc = glGetUniformLocation(shaderProgram[0], "viewPos");
	glUniform3f(uniformLoc, cameraPos[0], cameraPos[1], cameraPos[2]);
	uniformLoc = glGetUniformLocation(shaderProgram[0], "light.ambient");
	glUniform3f(uniformLoc, 0.2f, 0.2f, 0.2f);
	uniformLoc = glGetUniformLocation(shaderProgram[0], "light.diffuse");
	glUniform3f(uniformLoc, 0.5f, 0.5f, 0.5f);
	uniformLoc = glGetUniformLocation(shaderProgram[0], "light.specular");
	glUniform3f(uniformLoc, 1.0f, 1.0f, 1.0f);

	uniformLoc = glGetUniformLocation(shaderProgram[0], "material.shininess");
	glUniform1f(uniformLoc, 32.0f);

	uniformLoc = glGetUniformLocation(shaderProgram[0], "view");
	glUniformMatrix4fv(uniformLoc, 1, false, (float *)view);
	uniformLoc = glGetUniformLocation(shaderProgram[0], "projection");
	glUniformMatrix4fv(uniformLoc, 1, false, (float *)projection);

	glClear(GL_COLOR_BUFFER_BIT);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBindVertexArray(cubeVAO);
	// bind diffuse map
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture3);
	// remove specular map
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, 0);

	for (unsigned int i = 0; i < 5; i++)
	{
		drawCube(i);
	}

	// bind diffuse map
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture1);
	// bind specular map
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, texture2);

	for (unsigned int i = 5; i < 10; i++)
	{
		drawCube(i);
	}

	// fps counter
	++nbFrames;
	deltaTime = currentTime - lastFrame;
	lastFrame = currentTime;
	if (currentTime - lastTime >= 1.0)
	{
		fps = nbFrames;
		nbFrames = 0;
		lastTime += 1.0;
	}
}

int main(int argc, char **args)
{
	GLFWwindow *window;

	if (!glfwInit())
		return -1;

	window = glfwCreateWindow(screenWidth, screenHeight, "Textures test", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);
	glfwSwapInterval(0);
	gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
	glfwSetWindowSizeCallback(window, resizeCallback);
	glfwSetCursorPosCallback(window, mouseCallback);
	glfwSetScrollCallback(window, scrollCallback);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	init();
	while (!glfwWindowShouldClose(window))
	{
		render();

		glfwSwapBuffers(window);
		glfwPollEvents();
		processInput(window);
		sprintf(title, "fps: %d", fps);
		glfwSetWindowTitle(window, title);
	}

	return 0;
}