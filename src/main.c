#include <stdio.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <cglm/cglm.h>
#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#define CIMGUI_USE_GLFW
#define CIMGUI_USE_OPENGL3
#include <cimgui.h>
#include <cimgui_impl.h>

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
		"in float TextureID;\n"
		"uniform vec3 viewPos;\n"
		"uniform Material material;\n"
		"uniform sampler2DArray textures;\n"
		"uniform Light light;\n"
		"void main()\n"
		"{\n"
		" vec3 ambient=light.ambient*texture(textures,vec3(TexCoords, TextureID)).rgb;\n"
		" vec3 norm=normalize(Normal);\n"
		" vec3 lightDir=normalize(-light.direction);\n"
		" float diff=max(dot(norm,lightDir),0.);\n"
		" vec3 diffuse=light.diffuse*diff*texture(textures,vec3(TexCoords, TextureID)).rgb;\n"
		" vec3 viewDir=normalize(viewPos-FragPos);\n"
		" vec3 reflectDir=reflect(-lightDir,norm);\n"
		" float spec=pow(max(dot(viewDir,reflectDir),0.),material.shininess);\n"
		" vec3 result=ambient+diffuse;\n"
		" FragColor=vec4(result,1.);\n"
		"}";
const char *lightCasterVShader =
		"#version 330 core\n"
		"layout(location=0)in vec3 aPos;\n"
		"layout(location=1)in vec3 aNormal;\n"
		"layout(location=2)in vec2 aTexCoords;\n"
		"layout(location=3)in float aTextureID;\n"
		"out vec3 FragPos;\n"
		"out vec3 Normal;\n"
		"out vec2 TexCoords;\n"
		"out float TextureID;\n"
		"uniform mat4 model;\n"
		"uniform mat4 view;\n"
		"uniform mat4 projection;\n"
		"void main()\n"
		"{\n"
		" FragPos=vec3(model*vec4(aPos,1.));\n"
		" Normal=mat3(transpose(inverse(model)))*aNormal;\n"
		" TexCoords=aTexCoords;\n"
		" TextureID=aTextureID;\n"
		" gl_Position=projection*view*vec4(FragPos,1.);\n"
		"}";

unsigned int shaderProgram[2];
unsigned int textures[1024];
unsigned char *texturesData[1024];
unsigned int texture;
int textureCount = 0;
int shaderCount = 0;

double lastTime;
int nbFrames = 0;
int fps = 0;
char title[100];

#define SCREEN_WIDTH_INIT 800
#define SCREEN_HEIGHT_INIT 600

int screenWidth = SCREEN_WIDTH_INIT;
int screenHeight = SCREEN_HEIGHT_INIT;

void loadShader(unsigned int vertexShader, unsigned int fragmentShader)
{
	shaderProgram[shaderCount] = glCreateProgram();

	glAttachShader(shaderProgram[shaderCount], vertexShader);
	glAttachShader(shaderProgram[shaderCount], fragmentShader);
	glLinkProgram(shaderProgram[shaderCount]);

	glUseProgram(shaderProgram[shaderCount]);

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	++shaderCount;
}

void loadTexture(const char *imagePath)
{
	// load image
	int width, height, nrChannels;
	unsigned char *data = stbi_load(imagePath, &width, &height, &nrChannels, 4);
	if (data)
	{
		texturesData[textureCount++] = data;
	}
	else
	{
		puts("Failed to load texture");
	}
}

mat4 view = GLM_MAT4_IDENTITY_INIT;
mat4 projection;

vec3 cameraPos = {16.0f, 80.0f, 16.0f};
vec3 cameraFront = {0.0f, 0.0f, -1.0f};
vec3 cameraUp = {0.0f, 1.0f, 0.0f};
ivec2 lastChunk = {0, 0};
ivec2 cameraChunk = {0, 0};
vec2 cameraChunkPos = {0, 0};

float deltaTime = 0.0f; // time between current frame and last frame
float lastFrame = 0.0f;

bool cursorDisabled = true;
bool firstKey = true;

bool walked = false;

void processInput(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
	{
		if (firstKey)
		{
			firstKey = false;
		}
	}

	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_RELEASE)
	{
		if (!firstKey)
		{
			firstKey = true;

			if (cursorDisabled)
			{
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
				cursorDisabled = false;
			}
			else
			{
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
				cursorDisabled = true;
			}
		}
	}

	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	float cameraSpeed = (5 * deltaTime);
	vec3 scale;
	vec3 cross;
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
	{
		glm_vec3_scale(cameraFront, cameraSpeed, scale);
		glm_vec3_add(cameraPos, scale, cameraPos);
		walked = true;
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
	{
		glm_vec3_scale(cameraFront, -cameraSpeed, scale);
		glm_vec3_add(cameraPos, scale, cameraPos);
		walked = true;
	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
	{
		glm_cross(cameraFront, cameraUp, cross);
		glm_normalize(cross);
		glm_vec3_scale(cross, -cameraSpeed, scale);
		glm_vec3_add(cameraPos, scale, cameraPos);
		walked = true;
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
	{
		glm_cross(cameraFront, cameraUp, cross);
		glm_normalize(cross);
		glm_vec3_scale(cross, cameraSpeed, scale);
		glm_vec3_add(cameraPos, scale, cameraPos);
		walked = true;
	}
}

bool firstMouse = true;
float yaw = -90.0f;
float pitch = 0.0f;
float lastX = (float)SCREEN_HEIGHT_INIT / 2.0;
float lastY = (float)SCREEN_HEIGHT_INIT / 2.0;
float fov = 70.0f;

void resizeCallback(GLFWwindow *window, int width, int height)
{
	screenWidth = width;
	screenHeight = height;
	glViewport(0, 0, screenWidth, screenHeight);
	glm_perspective(glm_rad(fov), (float)width / (float)height, 0.1f, 100.0f, projection);
	glUseProgram(shaderProgram[0]);
	int uniformLoc = glGetUniformLocation(shaderProgram[0], "projection");
	glUniformMatrix4fv(uniformLoc, 1, false, (float *)projection);
}

void mouseCallback(GLFWwindow *window, double xposIn, double yposIn)
{
	if (!cursorDisabled)
		return;

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

static inline uint64_t
hashFunction(int8_t const *cstr, int len)
{
	uint64_t result = 14695981039346656037u;

	for (int i = 0; i < len; i++)
	{
		int64_t value = cstr[i];

		result ^= value;
		result *= 1099511628211u;
	}

	return result;
}

#define EXP 20

// Initialize all slots to an "empty" value (null)
#define HT_INIT \
	{             \
		{0}, 0      \
	}

typedef struct ht
{
	uint64_t ht[1 << EXP];
	int htIdx[1 << EXP];
	int32_t len;
} ht;

int32_t ht_lookup(uint64_t hash, int exp, int32_t idx)
{
	uint32_t mask = ((uint32_t)1 << exp) - 1;
	uint32_t step = (hash >> (64 - exp)) | 1;
	return (idx + step) & mask;
}

CGLM_INLINE
int32_t getHash(int posX, int posZ)
{
	int64_t pos = (posX & 0xFFFFFFFFll) | ((int64_t)posZ << 32);
	return hashFunction((int8_t *)&pos, sizeof(int64_t));
}

void addChunk(ht *t, const int posX, const int posZ, int index)
{
	int32_t h = getHash(posX, posZ);
	int32_t i = h;
	while (true)
	{
		i = ht_lookup(h, EXP, i);
		if (t->ht[i] == 0ull)
		{
			if ((uint32_t)t->len + 1 == (uint32_t)1 << EXP)
			{
				break;
			}
			t->len++;
			t->ht[i] = h;
			t->htIdx[i] = index;
			break;
		}
		else if (t->ht[i] == h)
		{
			break;
		}
	}
}

int hasChunk(ht *t, const int posX, const int posZ)
{
	int32_t h = getHash(posX, posZ);
	int32_t i = h;
	while (true)
	{
		i = ht_lookup(h, EXP, i);
		if (!t->ht[i])
		{
			return false;
		}
		else if (t->ht[i] == h)
		{
			return true;
		}
	}
}

int getChunkIdx(ht *t, const int posX, const int posZ)
{
	int32_t h = getHash(posX, posZ);
	int32_t i = h;
	while (true)
	{
		i = ht_lookup(h, EXP, i);
		if (!t->ht[i])
		{
			return -1;
		}
		else if (t->ht[i] == h)
		{
			return t->htIdx[i];
		}
	}
}

ht dimension = HT_INIT;

typedef int id[2];
typedef id chunk[16][256][16];

typedef bool FACES[6];

typedef struct meshCube
{
	ivec2 chunk;
	vec3 position;
	id ID;
	FACES faces;
} meshCube;

typedef struct chunkNode
{
	int index;
	int north;
	int south;
	int east;
	int west;
	int posX;
	int posZ;
	chunk chunk;
	int meshCount;
	int meshLimit;
	meshCube *meshCubes;
	int verticesBufferCount;
	int verticesBufferLimit;
	float *verticesBuffer;
	unsigned int VBO;
	unsigned int VAO;
} chunkNode;

int chunkCount = 0;
int chunkLimit = 1024;
chunkNode *chunks = NULL;

bool hasAir(chunkNode *chunkG, int x, int y, int z)
{
	if (y >= 256)
		return true;

	if (x >= 16)
	{
		if (chunkG->east >= 0)
		{
			x = 0;
			chunkG = &chunks[chunkG->east];
		}
		else
		{
			return false;
		}
	}
	else if (x < 0)
	{
		if (chunkG->west >= 0)
		{
			x = 15;
			chunkG = &chunks[chunkG->west];
		}
		else
		{
			return false;
		}
	}
	if (z >= 16)
	{
		if (chunkG->north >= 0)
		{
			z = 0;
			chunkG = &chunks[chunkG->north];
		}
		else
		{
			return false;
		}
	}
	else if (z < 0)
	{
		if (chunkG->south >= 0)
		{
			z = 15;
			chunkG = &chunks[chunkG->south];
		}
		else
		{
			return false;
		}
	}

	return chunkG->chunk[x][y][z][0] == 0;
}

void addFloat(chunkNode *chunkG, float number)
{
	chunkG->verticesBuffer[chunkG->verticesBufferCount++] = number;

	if (chunkG->verticesBufferCount + 1 >= chunkG->verticesBufferLimit - 1)
	{
		chunkG->verticesBufferLimit *= 2;
		chunkG->verticesBuffer = (float *)realloc(chunkG->verticesBuffer, sizeof(float) * chunkG->verticesBufferLimit);
	}
}

void addVertice(chunkNode *chunkG, float posX, float posY, float posZ, float normalX, float NormalY, float NormalZ, float textureCX, float textureCY, float textureID)
{
	addFloat(chunkG, posX);
	addFloat(chunkG, posY);
	addFloat(chunkG, posZ);
	addFloat(chunkG, normalX);
	addFloat(chunkG, NormalY);
	addFloat(chunkG, NormalZ);
	addFloat(chunkG, textureCX);
	addFloat(chunkG, textureCY);
	addFloat(chunkG, textureID);
}

void generateVertices(void)
{
	float postions[] = {};

	for (int i = 0; i <= chunkCount; i++)
	{
		chunkNode *chunkG = &chunks[i];

		for (int j = 0; j <= chunkG->meshCount; j++)
		{
			meshCube cube = chunkG->meshCubes[j];
			vec3 pos;
			glm_vec3_copy(cube.position, pos);
			glm_vec3_add(pos, (vec3){cube.chunk[0] * 16, 0, cube.chunk[1] * 16}, pos);
			int top = 0;
			int side = 0;
			int down = 0;
			if (cube.ID[0] == 1)
			{
				top = 2;
				side = 2;
				down = 2;
			}
			else if (cube.ID[0] == 2)
			{
				if (cube.ID[1] == 0)
				{
					top = 5;
					side = 5;
					down = 5;
				}
				else if (cube.ID[1] == 1)
				{
					top = 3;
					side = 4;
					down = 5;
				}
			}

			if (cube.faces[0])
			{
				addVertice(chunkG, pos[0] - .5, pos[1] + .5, pos[2] - .5, 0, 1, 0, 0, 0, top);
				addVertice(chunkG, pos[0] + .5, pos[1] + .5, pos[2] - .5, 0, 1, 0, 1, 0, top);
				addVertice(chunkG, pos[0] + .5, pos[1] + .5, pos[2] + .5, 0, 1, 0, 1, 1, top);
				addVertice(chunkG, pos[0] + .5, pos[1] + .5, pos[2] + .5, 0, 1, 0, 1, 1, top);
				addVertice(chunkG, pos[0] - .5, pos[1] + .5, pos[2] + .5, 0, 1, 0, 0, 1, top);
				addVertice(chunkG, pos[0] - .5, pos[1] + .5, pos[2] - .5, 0, 1, 0, 0, 0, top);
			}
			if (cube.faces[1] && pos[1] > 0)
			{
				addVertice(chunkG, pos[0] - .5, pos[1] - .5, pos[2] - .5, 0, -1, 0, 0, 0, down);
				addVertice(chunkG, pos[0] + .5, pos[1] - .5, pos[2] - .5, 0, -1, 0, 1, 0, down);
				addVertice(chunkG, pos[0] + .5, pos[1] - .5, pos[2] + .5, 0, -1, 0, 1, 1, down);
				addVertice(chunkG, pos[0] + .5, pos[1] - .5, pos[2] + .5, 0, -1, 0, 1, 1, down);
				addVertice(chunkG, pos[0] - .5, pos[1] - .5, pos[2] + .5, 0, -1, 0, 0, 1, down);
				addVertice(chunkG, pos[0] - .5, pos[1] - .5, pos[2] - .5, 0, -1, 0, 0, 0, down);
			}
			if (cube.faces[2])
			{
				addVertice(chunkG, pos[0] - .5, pos[1] - .5, pos[2] + .5, 0, 0, 1, 0, 0, side);
				addVertice(chunkG, pos[0] + .5, pos[1] - .5, pos[2] + .5, 0, 0, 1, 1, 0, side);
				addVertice(chunkG, pos[0] + .5, pos[1] + .5, pos[2] + .5, 0, 0, 1, 1, 1, side);
				addVertice(chunkG, pos[0] + .5, pos[1] + .5, pos[2] + .5, 0, 0, 1, 1, 1, side);
				addVertice(chunkG, pos[0] - .5, pos[1] + .5, pos[2] + .5, 0, 0, 1, 0, 1, side);
				addVertice(chunkG, pos[0] - .5, pos[1] - .5, pos[2] + .5, 0, 0, 1, 0, 0, side);
			}
			if (cube.faces[3])
			{
				addVertice(chunkG, pos[0] - .5, pos[1] - .5, pos[2] - .5, 0, 0, -1, 0, 0, side);
				addVertice(chunkG, pos[0] + .5, pos[1] - .5, pos[2] - .5, 0, 0, -1, 1, 0, side);
				addVertice(chunkG, pos[0] + .5, pos[1] + .5, pos[2] - .5, 0, 0, -1, 1, 1, side);
				addVertice(chunkG, pos[0] + .5, pos[1] + .5, pos[2] - .5, 0, 0, -1, 1, 1, side);
				addVertice(chunkG, pos[0] - .5, pos[1] + .5, pos[2] - .5, 0, 0, -1, 0, 1, side);
				addVertice(chunkG, pos[0] - .5, pos[1] - .5, pos[2] - .5, 0, 0, -1, 0, 0, side);
			}
			if (cube.faces[4])
			{
				addVertice(chunkG, pos[0] + .5, pos[1] - .5, pos[2] - .5, 1, 0, 0, 0, 0, side);
				addVertice(chunkG, pos[0] + .5, pos[1] - .5, pos[2] + .5, 1, 0, 0, 1, 0, side);
				addVertice(chunkG, pos[0] + .5, pos[1] + .5, pos[2] + .5, 1, 0, 0, 1, 1, side);
				addVertice(chunkG, pos[0] + .5, pos[1] + .5, pos[2] + .5, 1, 0, 0, 1, 1, side);
				addVertice(chunkG, pos[0] + .5, pos[1] + .5, pos[2] - .5, 1, 0, 0, 0, 1, side);
				addVertice(chunkG, pos[0] + .5, pos[1] - .5, pos[2] - .5, 1, 0, 0, 0, 0, side);
			}
			if (cube.faces[5])
			{
				addVertice(chunkG, pos[0] - .5, pos[1] - .5, pos[2] - .5, -1, 0, 0, 0, 0, side);
				addVertice(chunkG, pos[0] - .5, pos[1] - .5, pos[2] + .5, -1, 0, 0, 1, 0, side);
				addVertice(chunkG, pos[0] - .5, pos[1] + .5, pos[2] + .5, -1, 0, 0, 1, 1, side);
				addVertice(chunkG, pos[0] - .5, pos[1] + .5, pos[2] + .5, -1, 0, 0, 1, 1, side);
				addVertice(chunkG, pos[0] - .5, pos[1] + .5, pos[2] - .5, -1, 0, 0, 0, 1, side);
				addVertice(chunkG, pos[0] - .5, pos[1] - .5, pos[2] - .5, -1, 0, 0, 0, 0, side);
			}
		}
	}
}

void generateMesh()
{
	for (int i = 0; i < chunkCount; i++)
	{
		chunkNode *chunkG = &chunks[i];
		int posX = chunkG->posX;
		int posZ = chunkG->posZ;

		for (int x = 0; x < 16; x++)
		{
			for (int z = 0; z < 16; z++)
			{
				for (int y = 0; y < 256; y++)
				{
					meshCube cube = {
							.ID = {chunkG->chunk[x][y][z][0], chunkG->chunk[x][y][z][1]},
							.faces = {false, false, false, false, false, false},
							.chunk = {posX, posZ}};

					if (hasAir(chunkG, x, y, z))
						continue;

					if (hasAir(chunkG, x, y + 1, z))
						cube.faces[0] = true;
					if (hasAir(chunkG, x, y - 1, z))
						cube.faces[1] = true;
					if (hasAir(chunkG, x, y, z + 1))
						cube.faces[2] = true;
					if (hasAir(chunkG, x, y, z - 1))
						cube.faces[3] = true;
					if (hasAir(chunkG, x + 1, y, z))
						cube.faces[4] = true;
					if (hasAir(chunkG, x - 1, y, z))
						cube.faces[5] = true;

					for (int i = 0; i < 6; i++)
					{
						if (cube.faces[i])
						{
							vec3 position = {x, y, z};
							glm_vec3_copy(position, cube.position);
							chunkG->meshCubes[chunkG->meshCount++] = cube;

							if (chunkG->meshCount + 1 >= chunkG->meshLimit - 1)
							{
								chunkG->meshLimit *= 2;
								chunkG->meshCubes = (meshCube *)realloc(chunkG->meshCubes, sizeof(meshCube) * chunkG->meshLimit);
							}
							break;
						}
					}
				}
			}
		}
	}
}

float sine(int n)
{
	return 4 * sin(M_PI * (n - 4) / 8);
}

int map(int x, int y)
{
	return sine(x) + sine(y);
}

void generateChunk(chunkNode *chunkG, int posX, int posZ)
{
	if (chunkCount + 1 >= chunkLimit - 1)
	{
		chunkLimit *= 2;
		chunks = (chunkNode *)realloc(chunks, sizeof(chunkNode) * chunkLimit);
	}

	for (int x = 0; x < 16; x++)
	{
		for (int z = 0; z < 16; z++)
		{
			for (int y = 0; y < 256; y++)
			{
				if (y < map(x, z) + 55 + posX + posZ)
				{
					chunkG->chunk[x][y][z][0] = 1;
					chunkG->chunk[x][y][z][1] = 0;
				}
				else if (y < map(x, z) + 60 + posX + posZ)
				{
					chunkG->chunk[x][y][z][0] = 2;
					chunkG->chunk[x][y][z][1] = 0;
				}
				else if (y == map(x, z) + 60 + posX + posZ)
				{
					chunkG->chunk[x][y][z][0] = 2;
					chunkG->chunk[x][y][z][1] = 1;
				}
				else
				{
					chunkG->chunk[x][y][z][0] = 0;
					chunkG->chunk[x][y][z][1] = 0;
				}
			}
		}
	}
}

void generateChunkSides(ht *dimension)
{
	for (int i = 0; i < chunkCount; i++)
	{
		int sideIndex;
		chunkNode *chunkPtr = &chunks[i];
		int posX = chunkPtr->posX;
		int posZ = chunkPtr->posZ;
		if ((sideIndex = getChunkIdx(dimension, posX + 1, posZ)) >= 0)
		{
			chunkPtr->east = sideIndex;
			chunks[sideIndex].west = i;
		}
		if ((sideIndex = getChunkIdx(dimension, posX - 1, posZ)) >= 0)
		{
			chunkPtr->west = sideIndex;
			chunks[sideIndex].east = i;
		}
		if ((sideIndex = getChunkIdx(dimension, posX, posZ + 1)) >= 0)
		{
			chunkPtr->north = sideIndex;
			chunks[sideIndex].south = i;
		}
		if ((sideIndex = getChunkIdx(dimension, posX, posZ - 1)) >= 0)
		{
			chunkPtr->south = sideIndex;
			chunks[sideIndex].north = i;
		}
	}
}

chunkNode *generateChunkNode(const int posX, const int posZ)
{
	chunkNode *chunkPtr = &chunks[chunkCount++];

	chunkPtr->meshCount = 0;
	chunkPtr->verticesBufferCount = 0;
	chunkPtr->meshLimit = 1024;
	chunkPtr->verticesBufferLimit = 9 * 1024;

	chunkPtr->meshCubes = (meshCube *)malloc(chunkPtr->meshLimit * sizeof(meshCube));
	chunkPtr->verticesBuffer = (float *)malloc(chunkPtr->verticesBufferLimit * sizeof(float));

	chunkPtr->index = chunkCount - 1;
	chunkPtr->north = -1;
	chunkPtr->south = -1;
	chunkPtr->east = -1;
	chunkPtr->west = -1;
	chunkPtr->posX = posX;
	chunkPtr->posZ = posZ;

	generateChunk(chunkPtr, posX, posZ);

	return chunkPtr;
}

int viewDistance = 4;

void generateManyChunks(ht *dimension, int posX, int posZ)
{
	for (int i = -viewDistance + posX; i <= viewDistance + posX; i++)
	{
		for (int j = -viewDistance + posZ; j <= viewDistance + posZ; j++)
		{
			if (hasChunk(dimension, i, j)) continue;

			chunkNode *chunkPtr = generateChunkNode(i, j);
			int index = chunkPtr->index;
			addChunk(dimension, i, j, index);
		}
	}
}

void atualizeMovement(void)
{
	cameraChunk[0] = floor(cameraPos[0] / 16);
	cameraChunk[1] = floor(cameraPos[2] / 16);
	cameraChunkPos[0] = cameraPos[0] - cameraChunk[0];
	cameraChunkPos[1] = cameraPos[2] - cameraChunk[1];

	if (lastChunk[0] != cameraChunk[0] || lastChunk[1] != cameraChunk[1])
	{
		int lastChunkCount = chunkCount;
		printf("antes: %i\n", lastChunkCount);

		generateManyChunks(&dimension, cameraChunk[0], cameraChunk[1]);
		generateChunkSides(&dimension);
		generateMesh();
		generateVertices();

		lastChunk[0] = cameraChunk[0];
		lastChunk[1] = cameraChunk[1];

		for (int i = lastChunkCount; i <= chunkCount; i++)
		{
			printf("depois: %i\n", lastChunkCount);
			chunkNode *chunkG = &chunks[i];
			glGenVertexArrays(1, &chunkG->VAO);
			glGenBuffers(1, &chunkG->VBO);

			glBindBuffer(GL_ARRAY_BUFFER, chunkG->VBO);
			glBufferData(GL_ARRAY_BUFFER, chunkG->verticesBufferCount * sizeof(float), chunkG->verticesBuffer, GL_STATIC_DRAW);

			glBindVertexArray(chunkG->VAO);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void *)0);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void *)(3 * sizeof(float)));
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void *)(6 * sizeof(float)));
			glEnableVertexAttribArray(2);
			glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void *)(8 * sizeof(float)));
			glEnableVertexAttribArray(3);
		}
	}
}

ImGuiIO *ioptr;

void gui_init(GLFWwindow *window)
{
	/* Initialize CIMGUI */
	// GL 3.2 + GLSL 130
	const char *glsl_version = "#version 130";

	igCreateContext(NULL);

	// set docking
	ioptr = igGetIO();
	ioptr->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
// ioptr->ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;  // Enable Gamepad Controls
#ifdef IMGUI_HAS_DOCK
	ioptr->ConfigFlags |= ImGuiConfigFlags_DockingEnable;		// Enable Docking
	ioptr->ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // Enable Multi-Viewport / Platform Windows
#endif

	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init(glsl_version);

	igStyleColorsDark(NULL);
}

void gui_terminate(void)
{
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	igDestroyContext(NULL);
}

void gui_render(GLFWwindow *window)
{
	igRender();
	glfwMakeContextCurrent(window);
	ImGui_ImplOpenGL3_RenderDrawData(igGetDrawData());
}

int position[3] = {0, 0, 0};
// double currentTime;
// int toHash[2] = {0, 0};

void gui_update(GLFWwindow *window)
{
	// start imgui frame
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	igNewFrame();

	igBegin("Test", NULL, 0);
	igInputInt3("Position:", position, 0);
	igText(
			"Cube: %i:%i",
			chunks[0].chunk[position[0]][position[1]][position[2]][0],
			chunks[0].chunk[position[0]][position[1]][position[2]][1]);
	igText("Camera = x: %f, y: %f, z: %f", cameraPos[0], cameraPos[1], cameraPos[2]);
	igText("Camera chunk = x: %i, y: %i", cameraChunk[0], cameraChunk[1]);
	if (igButton("generate", (ImVec2){100, 30}))
	{
		atualizeMovement();
	}
	igEnd();

	gui_render(window);

#ifdef IMGUI_HAS_DOCK
	if (ioptr->ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
	{
		GLFWwindow *backup_current_window = glfwGetCurrentContext();
		igUpdatePlatformWindows();
		igRenderPlatformWindowsDefault(NULL, NULL);
		glfwMakeContextCurrent(backup_current_window);
	}
#endif
}

void init(void)
{
	chunks = (chunkNode *)malloc(chunkLimit * sizeof(chunkNode));
	// meshCubes = (meshCube *)malloc(meshLimit * sizeof(meshCube));
	// verticesBuffer = (float *)malloc(verticesBufferLimit * sizeof(float));

	generateManyChunks(&dimension, 0, 0);
	generateChunkSides(&dimension);
	generateMesh();
	generateVertices();

	glEnable(GL_DEPTH_TEST);

	for (int i = 0; i < chunkCount; i++)
	{
		chunkNode *chunkG = &chunks[i];
		glGenVertexArrays(1, &chunkG->VAO);
		glGenBuffers(1, &chunkG->VBO);

		glBindBuffer(GL_ARRAY_BUFFER, chunkG->VBO);
		glBufferData(GL_ARRAY_BUFFER, chunkG->verticesBufferCount * sizeof(float), chunkG->verticesBuffer, GL_STATIC_DRAW);

		glBindVertexArray(chunkG->VAO);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void *)0);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void *)(3 * sizeof(float)));
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void *)(6 * sizeof(float)));
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void *)(8 * sizeof(float)));
		glEnableVertexAttribArray(3);
	}

	unsigned int vertexShader;
	vertexShader = glCreateShader(GL_VERTEX_SHADER);

	glShaderSource(vertexShader, 1, &lightCasterVShader, NULL);
	glCompileShader(vertexShader);

	unsigned int fragmentShader;
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	glShaderSource(fragmentShader, 1, &lightCasterFShader, NULL);
	glCompileShader(fragmentShader);
	char infoLog[512];
	glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
	// printf("%i\n", glad_glGetError());
	// printf("%s\n", infoLog);

	loadShader(vertexShader, fragmentShader);

	vertexShader = glCreateShader(GL_VERTEX_SHADER);

	glShaderSource(vertexShader, 1, &cubeVShader, NULL);
	glCompileShader(vertexShader);

	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	glShaderSource(fragmentShader, 1, &cubeFShader, NULL);
	glCompileShader(fragmentShader);

	loadShader(vertexShader, fragmentShader);

	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D_ARRAY, texture);
	// set the texture wrapping parameters
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT); // set texture wrapping to GL_REPEAT (default wrapping method)
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT);
	// set texture filtering parameters
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	stbi_set_flip_vertically_on_load(true);
	loadTexture("../src/textures/default_box.png");
	loadTexture("../src/textures/default_iron_frame.png");
	loadTexture("../src/textures/default_stone.png");
	loadTexture("../src/textures/default_grass.png");
	loadTexture("../src/textures/default_dirt_with_grass.png");
	loadTexture("../src/textures/default_dirt.png");
	loadTexture("../src/textures/default_sand.png");

	void *data = malloc(4 * 16 * 16 * textureCount);
	for (size_t i = 0; i < textureCount; i++)
	{
		memcpy(data + 16 * 16 * i * 4, texturesData[i], 16 * 16 * 4);
		stbi_image_free(texturesData[i]);
	}

	glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, 16, 16, textureCount, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
	glGenerateMipmap(GL_TEXTURE_2D_ARRAY);
	free(data);

	glUseProgram(shaderProgram[0]);
	glUniform1i(glGetUniformLocation(shaderProgram[0], "textures"), 0);

	unsigned int uniformLoc = glGetUniformLocation(shaderProgram[0], "light.direction");
	glUniform3f(uniformLoc, -0.2f, -1.0f, -0.3f);
	uniformLoc = glGetUniformLocation(shaderProgram[0], "viewPos");
	glUniform3f(uniformLoc, cameraPos[0], cameraPos[1], cameraPos[2]);
	uniformLoc = glGetUniformLocation(shaderProgram[0], "light.ambient");
	glUniform3f(uniformLoc, 0.3f, 0.3f, 0.3f);
	uniformLoc = glGetUniformLocation(shaderProgram[0], "light.diffuse");
	glUniform3f(uniformLoc, 0.5f, 0.5f, 0.5f);
	uniformLoc = glGetUniformLocation(shaderProgram[0], "light.specular");
	glUniform3f(uniformLoc, 1.0f, 1.0f, 1.0f);

	uniformLoc = glGetUniformLocation(shaderProgram[0], "material.shininess");
	glUniform1f(uniformLoc, 32.0f);

	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);

	glm_translate(view, (vec3){0.0f, 0.0f, -3.0f});
	glm_perspective(glm_rad(fov), (float)screenWidth / (float)screenHeight, 0.1f, 200.0f, projection);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

double currentTime;

void render()
{
	currentTime = glfwGetTime();

	vec3 center;
	glm_vec3_add(cameraPos, cameraFront, center);
	glm_lookat(cameraPos, center, cameraUp, view);

	glUseProgram(shaderProgram[0]);

	unsigned int uniformLoc = glGetUniformLocation(shaderProgram[0], "view");
	glUniformMatrix4fv(uniformLoc, 1, false, (float *)view);
	uniformLoc = glGetUniformLocation(shaderProgram[0], "projection");
	glUniformMatrix4fv(uniformLoc, 1, false, (float *)projection);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBindVertexArray(cubeVAO);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D_ARRAY, texture);

	mat4 model = GLM_MAT4_IDENTITY_INIT;
	glUseProgram(shaderProgram[0]);
	unsigned int transformLoc = glGetUniformLocation(shaderProgram[0], "model");
	glUniformMatrix4fv(transformLoc, 1, false, (float *)model);
	for (int i = 0; i <= chunkCount; i++)
	{
		glBindVertexArray(chunks[i].VAO);
		glDrawArrays(GL_TRIANGLES, 0, chunks[i].verticesBufferCount / 8);
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

	if (walked)
	{
		atualizeMovement();
		walked = false;
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
	gui_init(window);
	while (!glfwWindowShouldClose(window))
	{
		render();
		glFlush();
		gui_update(window);

		glfwSwapBuffers(window);
		glfwPollEvents();
		processInput(window);
		sprintf(title, "fps: %d", fps);
		glfwSetWindowTitle(window, title);
	}

	glfwTerminate();
	gui_terminate();
	return 0;
}